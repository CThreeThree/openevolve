"""
Run evaluations as C3 jobs (evaluator.backend: "c3").

`C3RemoteEvaluate` is a drop-in replacement for the user's `evaluate(program_path)`: the
Evaluator calls it from the same thread executor, inside the same process pool, with the same
timeout and retry handling. Instead of importing the evaluation file locally, it packs the
candidate, the evaluation file (plus `evaluator.c3_include` files) and a small harness into a
tar.zst bundle, submits it to C3 with the caller's `C3_API_KEY`, waits for the job and returns
the metrics and artifacts the harness wrote to `results/result.json`.

If a `requirements.txt` sits beside the evaluator (or `evaluator.c3_requirements` names one),
the job installs it into a venv with `uv` first; job machines have only a bare system Python.

A candidate that crashes is returned as `EvaluationResult(metrics={"error": 0.0}, artifacts=
{"traceback": ...})`, so its feedback reaches the next prompt without being retried.
Infrastructure and setup failures (provisioning, cancelled jobs, API errors, requirements that
fail to install, an evaluator that cannot be imported) raise, so the Evaluator's
normal `max_retries` applies to them.

Requires `zstandard` (pip install "openevolve[c3]") unless running on Python 3.14+.
"""

import io
import json
import logging
import os
import tarfile
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from openevolve.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)

USER_AGENT = "openevolve-c3/1"  # Cloudflare rejects the default Python-urllib User-Agent
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELED", "TIMED_OUT"}
RESULT_LIMIT = 1024 * 1024

# Runs inside the C3 job: load evaluator.py, score program.py, write results/result.json.
HARNESS = r"""import json, math, os, pathlib, sys, traceback
sys.path.insert(0, os.getcwd())
out = pathlib.Path("results"); out.mkdir(exist_ok=True)
def emit(doc): (out / "result.json").write_text(json.dumps(doc))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluation_module", "evaluator.py")
    module = importlib.util.module_from_spec(spec); sys.modules["evaluation_module"] = module
    spec.loader.exec_module(module)
except Exception as exc:  # the evaluator itself cannot start: an environment problem, not the candidate
    text = traceback.format_exc()
    sys.stderr.write(text)
    emit({"metrics": None, "stage": "setup", "error": f"{type(exc).__name__}: {exc}", "artifacts": {"traceback": text[-8192:]}})
    sys.exit(1)
try:
    result = module.evaluate("program.py", **json.loads(pathlib.Path("evaluator_args.json").read_text()))
    if hasattr(result, "metrics"):
        metrics, artifacts = dict(result.metrics), dict(getattr(result, "artifacts", {}) or {})
    else:
        metrics = dict(result); artifacts = metrics.pop("artifacts", {}) or {}
    metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
    emit({"metrics": metrics, "artifacts": {str(k): (v if isinstance(v, str) else repr(v))[:65536] for k, v in artifacts.items()}})
except Exception as exc:
    text = traceback.format_exc()
    sys.stderr.write(text)
    emit({"metrics": None, "error": f"{type(exc).__name__}: {exc}", "artifacts": {"traceback": text[-8192:]}})
    sys.exit(1)
"""
RUN_SCRIPT = """#!/bin/sh
set -eu
PY=python3
if [ -f requirements.txt ]; then
  # Job machines have uv and a bare system Python: install the evaluator's dependencies first.
  if ! { uv venv -q .venv && uv pip install -q --python .venv/bin/python -r requirements.txt; } > setup.log 2>&1; then
    mkdir -p results
    python3 -c 'import json, pathlib; pathlib.Path("results/result.json").write_text(json.dumps({"metrics": None, "stage": "setup", "error": "installing requirements.txt failed", "artifacts": {"setup_log": pathlib.Path("setup.log").read_text()[-8192:]}}))'
    cat setup.log >&2
    exit 1
  fi
  PY=.venv/bin/python
fi
exec "$PY" evaluate_c3.py
"""
MANIFEST = json.dumps({"version": "1", "project_root": ".", "submission_script_path": "run.sh"})


def _zstd(data: bytes) -> bytes:
    try:
        from compression import zstd  # Python 3.14+

        return zstd.compress(data)
    except ImportError:
        try:
            import zstandard
        except ImportError:
            raise RuntimeError(
                'evaluator.backend "c3" needs zstandard: pip install "openevolve[c3]"'
            ) from None
        return zstandard.ZstdCompressor().compress(data)


def build_bundle(
    program: str, evaluator: str, evaluator_args: Dict[str, Any], include: Dict[str, bytes]
) -> bytes:
    """The job bundle: everything under workspace/, plus the manifest the C3 agent reads."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "evaluation_result.py"), "rb") as handle:
        evaluation_result = handle.read()  # lets evaluators import openevolve.evaluation_result
    files = {f"workspace/{name}": data for name, data in include.items()}
    files.update(
        {
            "workspace/program.py": program.encode(),
            "workspace/evaluator.py": evaluator.encode(),
            "workspace/evaluator_args.json": json.dumps(evaluator_args or {}).encode(),
            "workspace/evaluate_c3.py": HARNESS.encode(),
            "workspace/run.sh": RUN_SCRIPT.encode(),
            "workspace/openevolve/__init__.py": b"",
            "workspace/openevolve/evaluation_result.py": evaluation_result,
            "c3/manifest.json": MANIFEST.encode(),
        }
    )
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), 0
            info.mode = 0o755 if name.endswith("run.sh") else 0o644
            tar.addfile(info, io.BytesIO(data))
    return _zstd(raw.getvalue())


class C3Error(RuntimeError):
    pass


class C3RemoteEvaluate:
    """Callable with the user's evaluate() signature; evaluates on a C3 job."""

    def __init__(self, evaluation_file: str, config):
        self.evaluation_file = evaluation_file
        self.config = config
        self.api_key = os.environ.get("C3_API_KEY")
        if not self.api_key:
            raise ValueError('evaluator.backend "c3" needs the C3_API_KEY environment variable')
        self.endpoint = (
            config.c3_api_endpoint
            or os.environ.get("C3_API_ENDPOINT")
            or "https://api.cthree.cloud"
        ).rstrip("/")
        with open(evaluation_file) as handle:
            self.evaluator_source = handle.read()
        base = os.path.dirname(os.path.abspath(evaluation_file))
        self.include = {}
        requirements = config.c3_requirements or "requirements.txt"
        if os.path.isfile(os.path.join(base, requirements)):
            with open(os.path.join(base, requirements), "rb") as handle:
                self.include["requirements.txt"] = handle.read()
        elif config.c3_requirements:
            raise ValueError(f"c3_requirements not found: {os.path.join(base, requirements)}")
        for name in config.c3_include or []:
            if os.path.isabs(name) or ".." in name.split("/"):
                raise ValueError(f"c3_include entries must be relative paths inside {base}: {name}")
            with open(os.path.join(base, name), "rb") as handle:
                self.include[name] = handle.read()

    # ----- HTTP -------------------------------------------------------------------------
    def _call(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        request = urllib.request.Request(
            self.endpoint + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise C3Error(
                f"C3 API {method} {path.split('?')[0]} -> {error.code}: "
                f"{error.read()[:300].decode(errors='replace')}"
            ) from None
        return json.loads(raw) if raw else {}

    def _storage(self, method: str, url: str, data: Optional[bytes] = None) -> bytes:
        request = urllib.request.Request(
            url,
            method=method,
            data=data,
            headers={
                "User-Agent": USER_AGENT,
                **({"Content-Type": "application/zstd"} if data else {}),
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # no C3 key to storage
            return response.read(RESULT_LIMIT + 1)

    # ----- one evaluation ----------------------------------------------------------------
    def __call__(self, program_path: str) -> EvaluationResult:
        with open(program_path) as handle:
            program = handle.read()
        bundle = build_bundle(
            program, self.evaluator_source, self.config.evaluator_args or {}, self.include
        )
        upload = self._call(
            "POST", "/v2/storage/presign-upload", {"content_type": "application/zstd"}
        )
        if upload.get("url"):
            self._storage("PUT", upload["url"], bundle)
        job = {
            "bundle_uri": "c3://" + upload["key"],
            "submission_script_path": "run.sh",
            "output_folders": ["results"],
            "hardware_profile": self.config.c3_hardware,
            "timeout_seconds": int(self.config.timeout),
            "project": self.config.c3_project,
            "job_name": f"openevolve-{os.path.basename(program_path)}",
        }
        if self.config.c3_provider:
            job["provider"] = self.config.c3_provider
        job_id = self._call("POST", "/v2/jobs", job)["id"]
        logger.info(f"Evaluating {program_path} on C3 job {job_id}")
        try:
            state = self._wait(job_id)
        except BaseException:
            self._cancel(job_id)  # deadline, Ctrl-C or error: never leave a paid job running
            raise
        return self._collect(job_id, state)

    def _wait(self, job_id: str) -> dict:
        queue_deadline = time.monotonic() + self.config.c3_queue_timeout
        run_deadline = None
        while True:
            state = self._call("GET", f"/v2/jobs/{job_id}")
            status = state.get("status")
            if status in TERMINAL:
                return state
            now = time.monotonic()
            if status == "RUNNING" or state.get("started_at"):
                run_deadline = run_deadline or now + self.config.timeout + 120
                if now > run_deadline:
                    raise TimeoutError(f"C3 job {job_id} exceeded {self.config.timeout}s")
            elif now > queue_deadline:
                raise TimeoutError(
                    f"C3 job {job_id} did not start within {self.config.c3_queue_timeout}s"
                )
            time.sleep(3)

    def _collect(self, job_id: str, state: dict) -> EvaluationResult:
        status, attempt = state["status"], (state.get("latest_attempt") or {}).get("id")
        reported = self._result(attempt) if attempt else None
        if status == "SUCCEEDED" and reported and isinstance(reported.get("metrics"), dict):
            return EvaluationResult(
                metrics=reported["metrics"], artifacts=reported.get("artifacts") or {}
            )
        failure = state.get("failure") or {}
        if reported and reported.get("stage") == "setup":
            detail = reported.get("artifacts") or {}
            raise C3Error(
                f"evaluator could not start on C3 job {job_id}: {reported.get('error')}\n"
                + (detail.get("traceback") or detail.get("setup_log") or "")[-2000:]
            )
        ran = failure.get("stage") == "run" or (reported and reported.get("metrics") is None)
        if not ran:  # provisioning/cancellation/etc.: let the Evaluator's retries handle it
            raise C3Error(
                f"C3 job {job_id} {status}: {failure.get('message') or failure.get('code') or ''}"
            )
        artifacts = (reported or {}).get("artifacts") or {"log_tail": self._log_tail(job_id)}
        artifacts.update(failure_stage="evaluation", c3_job=job_id)
        return EvaluationResult(metrics={"error": 0.0}, artifacts=artifacts)

    def _result(self, attempt_id: str) -> Optional[dict]:
        artifacts = self._call("GET", f"/v2/attempts/{attempt_id}/artifacts").get("artifacts") or []
        entry = next(
            (a for a in artifacts if str(a.get("path", "")).endswith("results/result.json")), None
        )
        if not entry:
            return None
        signed = self._call(
            "GET", "/v2/storage/presign-download/" + entry["object_uri"].removeprefix("c3://")
        )
        try:
            return json.loads(self._storage("GET", signed["url"])[:RESULT_LIMIT])
        except ValueError:
            return None

    def _log_tail(self, job_id: str) -> str:
        try:
            page = self._call("GET", f"/v2/jobs/{job_id}/logs?limit=500&stream=stdout,setup")
            return "\n".join(str(e.get("line", "")) for e in page.get("entries") or [])[-4096:]
        except C3Error:
            return ""

    def _cancel(self, job_id: str) -> None:
        try:
            self._call("POST", f"/v2/jobs/{job_id}/cancel", {})
            logger.warning(f"Cancelled C3 job {job_id}")
        except Exception as exc:
            logger.error(f"Could not cancel C3 job {job_id}: {exc}")
