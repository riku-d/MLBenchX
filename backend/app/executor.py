"""Run user-supplied Python without letting it out.

The editor in this app exists so people can change the pipeline. That means the backend
executes code a user wrote, which is the single most dangerous thing this application
does, and the defences here are what make the feature acceptable rather than reckless.

What is enforced
----------------
Every run gets a fresh directory containing only its own input, and a child process that:

* runs under an **OS sandbox profile** (macOS ``sandbox-exec``) that denies the network
  outright and denies writes anywhere except that directory — so even a successful
  ``rm -rf`` against the project is refused by the kernel, not by our hopefulness;
* carries **resource limits** — address space, CPU seconds, output file size, and a
  process cap so a fork bomb cannot spawn;
* is killed after a **wall-clock timeout**, process group and all, because a program
  that blocks on something the CPU limit never notices would otherwise hang forever;
* gets a **scrubbed environment**: no inherited credentials, no proxy settings, no
  ``PATH`` to the user's tools.

What is not
-----------
On a platform without ``sandbox-exec`` the OS layer is unavailable and only the process
limits apply. That is weaker — a determined script could read files the server user can
read, and could reach the network — so :func:`sandbox_status` reports which layer is
active and the workspace shows it. The honest production answer is a container with no
network namespace, and the interface here is shaped so that swapping one in means
replacing :func:`_command` alone.

None of this makes running someone else's code safe in general. It makes it safe enough
for a tool whose user is running *their own* code against *their own* data on their own
machine, which is what this is.
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

from app.config import (
    EXECUTION_CPU_SECONDS,
    EXECUTION_MAX_OUTPUT_BYTES,
    EXECUTION_MEMORY_BYTES,
    EXECUTION_TIMEOUT_SECONDS,
)

__all__ = [
    "ExecutionResult",
    "ExecutionError",
    "run_pipeline",
    "sandbox_status",
]

_SANDBOX_BINARY = "/usr/bin/sandbox-exec"

#: Deny everything, then allow the minimum a pandas/sklearn script needs.
#:
#: ``file-read*`` is broad on purpose: the interpreter, its standard library and every
#: installed package have to be readable, and enumerating those paths would break on the
#: next environment rebuild. Reading is also the weaker half of the risk — the script
#: has no network to send anything to, and nothing it can write outside its own
#: directory. ``mach-lookup`` and ``sysctl-read`` are required by CPython itself;
#: without them the interpreter will not start.
_PROFILE = """(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow sysctl-read)
(allow mach-lookup)
(allow file-read*)
(allow file-ioctl)
(allow file-write* (subpath "{workdir}"))
(allow file-write-data (literal "/dev/null"))
(deny network*)
"""


class ExecutionError(Exception):
    """The pipeline could not be run at all, as opposed to running and failing."""


@dataclass
class ExecutionResult:
    """What one run produced."""

    ok: bool
    stdout: str
    stderr: str
    duration_ms: int
    exit_code: int
    #: The processed frame, when the script wrote one.
    frame: pd.DataFrame | None = None
    #: The step report the generated script writes. Empty when the user removed the
    #: reporting calls, which is allowed.
    report: dict[str, Any] = field(default_factory=dict)
    #: Structured failure, for the error panel.
    error: dict[str, Any] | None = None
    #: Which isolation layer actually applied.
    sandbox: str = "none"

    @property
    def logs(self) -> list[str]:
        """Console output as lines, for the log pane."""
        combined = self.stdout + ("\n" + self.stderr if self.stderr else "")
        return [line for line in combined.splitlines() if line.strip()]


def sandbox_status() -> dict[str, Any]:
    """Which isolation is available here, in terms the interface can show a user."""
    if sys.platform == "darwin" and Path(_SANDBOX_BINARY).exists():
        return {
            "level": "os_sandbox",
            "label": "OS sandbox",
            "detail": (
                "Your code runs in a kernel-enforced sandbox with no network access and "
                "no ability to write outside its own scratch directory, under memory "
                "and CPU limits."
            ),
            "network_blocked": True,
            "filesystem_isolated": True,
        }
    return {
        "level": "limits_only",
        "label": "Process limits only",
        "detail": (
            "This platform has no OS sandbox available, so your code runs in a separate "
            "process with memory, CPU and time limits, but it can reach the network and "
            "read files this server can read. Run untrusted code in a container instead."
        ),
        "network_blocked": False,
        "filesystem_isolated": False,
    }


def _limits() -> None:
    """Applied in the child between fork and exec.

    Each limit is best-effort and independently guarded. A platform that refuses one of
    them should lose that single layer, not the whole run: raising here aborts the
    process before ``exec``, which would take the OS sandbox down with it and leave the
    user unable to run anything at all.

    ``RLIMIT_AS`` and ``RLIMIT_DATA`` are attempted and will succeed on Linux. macOS
    refuses to lower either at any value — even far above what the process uses — so on
    macOS the memory ceiling is enforced by :func:`_watch_memory` from the parent
    instead. A tutorial-standard ``setrlimit(RLIMIT_AS, ...)`` here would look like a
    memory cap and be nothing of the kind.
    """
    for which, value in (
        (resource.RLIMIT_AS, EXECUTION_MEMORY_BYTES),
        (resource.RLIMIT_DATA, EXECUTION_MEMORY_BYTES),
        (resource.RLIMIT_CPU, EXECUTION_CPU_SECONDS),
        (resource.RLIMIT_FSIZE, EXECUTION_MAX_OUTPUT_BYTES),
        (resource.RLIMIT_NPROC, 256),
    ):
        try:
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError):
            continue

    # A new process group, so a timeout or a memory breach can kill the script and
    # everything it spawned rather than only the process we hold a handle to.
    try:
        os.setsid()
    except OSError:
        pass


def _human_bytes(count: int) -> str:
    """A byte count as a person would write it. Integer GB alone renders 512 MB as 0."""
    if count >= 1024 ** 3:
        gigabytes = count / 1024 ** 3
        return f"{gigabytes:.1f}".rstrip("0").rstrip(".") + " GB"
    return f"{count // (1024 ** 2)} MB"


def _kill_group(process: subprocess.Popen) -> None:
    """Kill the child and every process it started."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


def _group_memory(process: psutil.Process) -> int:
    """Resident bytes used by the child and its descendants."""
    total = 0
    try:
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total


def _watch_memory(process: subprocess.Popen, ceiling: int, breached: threading.Event) -> None:
    """Kill the run if it grows past ``ceiling``.

    Polls rather than waits, because there is no portable notification for "this process
    got too big". A tenth of a second is far finer than the allocation rate that would
    put a machine in trouble, and costs nothing next to the pipeline itself.
    """
    try:
        handle = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    while process.poll() is None:
        if _group_memory(handle) > ceiling:
            breached.set()
            _kill_group(process)
            return
        time.sleep(0.1)


def _command(script: Path, workdir: Path) -> tuple[list[str], str]:
    """The argv to run, and the name of the isolation layer it applies."""
    python = sys.executable
    if sys.platform == "darwin" and Path(_SANDBOX_BINARY).exists():
        profile = workdir / "sandbox.sb"
        # The profile must name the *resolved* path: /tmp is a symlink on macOS and the
        # sandbox matches on the real one, so an unresolved path silently allows nothing.
        profile.write_text(_PROFILE.format(workdir=str(workdir.resolve())))
        return [_SANDBOX_BINARY, "-f", str(profile), python, str(script)], "os_sandbox"
    return [python, str(script)], "limits_only"


def _library_path() -> str:
    """The import paths the child needs, and only those.

    The environment is otherwise built from nothing, which means the child cannot fall
    back on whatever the parent happened to have. But it still has to import pandas and
    scikit-learn, and those can live across several site-packages directories — a
    virtualenv's own, the user site, and the interpreter's. Blanking ``PYTHONPATH``
    loses whichever of those is not the interpreter's, so the paths are passed through
    explicitly rather than inherited implicitly.
    """
    return os.pathsep.join(p for p in sys.path if p and "site-packages" in p)


def _environment(workdir: Path) -> dict[str, str]:
    """A minimal environment. Nothing is inherited that does not have to be."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": _library_path(),
        # Numerical libraries otherwise start a thread pool per core, which collides
        # with the process cap and buys nothing on data this size.
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "JOBLIB_MULTIPROCESSING": "0",
        "MLBX_INPUT": str(workdir / "input.csv"),
        "MLBX_OUTPUT": str(workdir / "processed.csv"),
        "MLBX_REPORT": str(workdir / "report.json"),
    }


#: Maps an exception to something the user can act on. Ordered: the first match wins,
#: so put the specific patterns above the general ones.
_DIAGNOSES: tuple[tuple[str, str, str], ...] = (
    (
        r"could not convert string to float: ['\"](.+?)['\"]",
        "A column you are treating as numeric contains the text {0!r}.",
        "Either exclude that column, or leave Data Type Detection on so it is "
        "recognised as a category.",
    ),
    (
        r"Input (?:X )?contains NaN",
        "A step received blank values it cannot handle.",
        "Turn Missing Value Treatment on, or move it above this step.",
    ),
    (
        r"KeyError: ['\"](.+?)['\"]",
        "The column {0!r} does not exist at this point in the pipeline.",
        "An earlier step may have renamed or removed it — check the step order, or the "
        "spelling against the dataset preview.",
    ),
    (
        r"Found unknown categories",
        "The held-out rows contain a category the training rows never had.",
        "Set the encoder's handle_unknown to 'ignore'.",
    ),
    (
        r"(?:n_splits|cv)=(\d+).+?(?:minimum|fewer)",
        "A class has fewer rows than the number of cross-validation folds.",
        "Lower the number of folds, or use an encoder that does not read the target.",
    ),
    (
        r"MemoryError|Unable to allocate",
        "The pipeline tried to allocate more memory than the sandbox allows.",
        "One-hot encoding a high-cardinality column is the usual cause — lower the "
        "one-hot threshold so crowded columns use frequency encoding instead.",
    ),
    (
        r"ValueError: (?:Expected 2D array|Found array with)",
        "A transformer was handed data in the wrong shape.",
        "Most often a single column passed as a Series where a frame was expected — "
        "use df[['column']] rather than df['column'].",
    ),
    (
        r"SyntaxError",
        "The script is not valid Python.",
        "The line number below points at the problem; a missing bracket or colon on the "
        "line above is the usual cause.",
    ),
)


def _diagnose(stderr: str) -> dict[str, Any] | None:
    """Turn a traceback into something worth showing.

    'Something went wrong' is not an error message. The user is looking at their own
    code in an editor two panels away, so what they need is the line, the exception, and
    a sentence about what usually causes it.
    """
    if not stderr.strip():
        return None

    # The last frame naming the script is where the user's code actually failed;
    # frames below it are inside pandas or sklearn and are not where they should look.
    lines = [int(m) for m in re.findall(r'File "[^"]*pipeline\.py", line (\d+)', stderr)]
    syntax = re.search(r'File "[^"]*pipeline\.py", line (\d+)', stderr)
    line_number = lines[-1] if lines else (int(syntax.group(1)) if syntax else None)

    final = stderr.strip().splitlines()[-1].strip()
    exception_type = final.split(":", 1)[0].strip() if ":" in final else final
    message = final.split(":", 1)[1].strip() if ":" in final else ""

    cause = suggestion = None
    for pattern, cause_template, advice in _DIAGNOSES:
        match = re.search(pattern, stderr)
        if match:
            cause = cause_template.format(*match.groups()) if match.groups() else cause_template
            suggestion = advice
            break

    return {
        "type": exception_type,
        "message": message or final,
        "line": line_number,
        "cause": cause,
        "suggestion": suggestion,
        "traceback": stderr.strip()[-4000:],
    }


def run_pipeline(code: str, dataset_csv: bytes, timeout: int = EXECUTION_TIMEOUT_SECONDS) -> ExecutionResult:
    """Execute ``code`` against ``dataset_csv`` and bring back what it produced.

    Args:
        code: the Python to run — generated, user-edited, or a mix.
        dataset_csv: the dataset, as the bytes the user uploaded.
        timeout: wall-clock seconds before the process group is killed.

    Returns:
        An :class:`ExecutionResult`. A script that runs and fails is a *result* with
        ``ok=False`` and a populated ``error``, not an exception — a failing pipeline is
        an ordinary outcome the user needs to see, not an incident.

    Raises:
        ExecutionError: only when the run could not be attempted.
    """
    workdir = Path(tempfile.mkdtemp(prefix="mlbx-run-"))
    started = time.perf_counter()
    try:
        script = workdir / "pipeline.py"
        script.write_text(code, encoding="utf-8")
        (workdir / "input.csv").write_bytes(dataset_csv)

        argv, sandbox = _command(script, workdir)
        breached = threading.Event()
        process = subprocess.Popen(
            argv,
            cwd=workdir,
            env=_environment(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_limits,
        )
        monitor = threading.Thread(
            target=_watch_memory,
            args=(process, EXECUTION_MEMORY_BYTES, breached),
            daemon=True,
        )
        monitor.start()

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            stdout, stderr = process.communicate()
            timed_out = True
        exit_code = process.returncode
        monitor.join(timeout=1.0)

        duration_ms = round((time.perf_counter() - started) * 1000)

        if breached.is_set():
            return ExecutionResult(
                ok=False, stdout=stdout, stderr=stderr, duration_ms=duration_ms,
                exit_code=exit_code, sandbox=sandbox,
                error={
                    "type": "MemoryLimit",
                    "message": (
                        f"The pipeline used more than {_human_bytes(EXECUTION_MEMORY_BYTES)} "
                        "of memory and was stopped."
                    ),
                    "line": None,
                    "cause": "A step allocated far more memory than this dataset needs.",
                    "suggestion": (
                        "One-hot encoding a column with thousands of distinct values is "
                        "the usual cause — lower the one-hot threshold so crowded "
                        "columns use frequency encoding instead."
                    ),
                    "traceback": stderr.strip()[-4000:],
                },
            )

        if timed_out:
            return ExecutionResult(
                ok=False, stdout=stdout, stderr=stderr, duration_ms=duration_ms,
                exit_code=exit_code, sandbox=sandbox,
                error={
                    "type": "Timeout",
                    "message": f"The pipeline ran for longer than {timeout} seconds and was stopped.",
                    "line": None,
                    "cause": "A step is taking far longer than this dataset should need.",
                    "suggestion": (
                        "An unbounded loop, or one-hot encoding a column with thousands "
                        "of distinct values, are the usual causes."
                    ),
                    "traceback": stderr.strip()[-4000:],
                },
            )

        if exit_code != 0:
            return ExecutionResult(
                ok=False, stdout=stdout, stderr=stderr, duration_ms=duration_ms,
                exit_code=exit_code, sandbox=sandbox,
                error=_diagnose(stderr) or {
                    "type": "ExitCode",
                    "message": f"The pipeline exited with status {exit_code} and no traceback.",
                    "line": None,
                    "cause": (
                        "Usually the sandbox stopping the process — a memory or CPU "
                        "limit, or an attempt to reach the network."
                    ),
                    "suggestion": "Check the logs below for the last step that ran.",
                    "traceback": stderr.strip()[-4000:],
                },
            )

        output = workdir / "processed.csv"
        if not output.is_file():
            return ExecutionResult(
                ok=False, stdout=stdout, stderr=stderr, duration_ms=duration_ms,
                exit_code=exit_code, sandbox=sandbox,
                error={
                    "type": "NoOutput",
                    "message": "The pipeline finished without writing a processed dataset.",
                    "line": None,
                    "cause": "Nothing was written to OUTPUT_PATH.",
                    "suggestion": (
                        "Keep the final `df.to_csv(OUTPUT_PATH, index=False)` line — the "
                        "app reads the result from there."
                    ),
                    "traceback": "",
                },
            )

        frame = pd.read_csv(output)
        report_path = workdir / "report.json"
        report: dict[str, Any] = {}
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text())
            except json.JSONDecodeError:
                # The user is free to rewrite the reporting; unreadable is not fatal.
                report = {}

        return ExecutionResult(
            ok=True, stdout=stdout, stderr=stderr, duration_ms=duration_ms,
            exit_code=0, frame=frame, report=report, sandbox=sandbox,
        )
    except OSError as exc:
        raise ExecutionError(f"could not start the pipeline process: {exc}") from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
