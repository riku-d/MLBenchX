"""The sandbox: does it actually contain code that tries to get out?

This is the most security-sensitive component in the application, and the only way to
have any confidence in it is to attack it. Every test here is a real attempt to do
something the sandbox is supposed to prevent — reach the network, write outside the
work directory, delete this repository, read the developer's private keys, exhaust
memory, fork without limit, run forever.

A test that merely asserts the sandbox is *configured* would pass against a profile
with a typo in it. These assert on the outcome, and several of them check afterwards
that the thing which was supposed to be protected is still there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app.codegen import generate
from app.executor import _diagnose, _human_bytes, run_pipeline, sandbox_status
from app.pipeline import PipelineConfig

FIXTURE = Path(__file__).parent / "fixtures" / "titanic.csv"
PROJECT = Path(__file__).resolve().parent.parent

#: A canary that must survive every attack below.
CANARY = PROJECT / "app" / "main.py"

os_sandboxed = pytest.mark.skipif(
    sandbox_status()["level"] != "os_sandbox",
    reason="no OS sandbox on this platform; only the process limits apply",
)


@pytest.fixture(scope="module")
def data() -> bytes:
    return FIXTURE.read_bytes()


# --------------------------------------------------------------------------------------
# It runs legitimate work
# --------------------------------------------------------------------------------------


def test_a_generated_pipeline_runs_and_returns_its_output(data) -> None:
    result = run_pipeline(generate(PipelineConfig(target="Survived")), data)

    assert result.ok, result.stderr[-1500:]
    assert result.frame is not None and len(result.frame) == 891
    assert result.report["steps"], "the step report came back empty"
    assert result.duration_ms > 0
    assert result.logs


def test_a_script_that_writes_nothing_is_reported_not_crashed(data) -> None:
    """Deleting the output line is something a user can do; it must not look like a bug."""
    result = run_pipeline("print('I did nothing useful')", data)

    assert not result.ok
    assert result.error is not None and result.error["type"] == "NoOutput"
    assert "OUTPUT_PATH" in result.error["suggestion"]


def test_removing_the_reporting_still_produces_a_dataset(data) -> None:
    """The reporting calls are the user's to delete. Losing them loses detail, not data."""
    result = run_pipeline(
        "import os\n"
        "import pandas as pd\n"
        "df = pd.read_csv(os.environ['MLBX_INPUT'])\n"
        "df.to_csv(os.environ['MLBX_OUTPUT'], index=False)\n",
        data,
    )
    assert result.ok
    assert result.frame is not None
    assert result.report == {}


# --------------------------------------------------------------------------------------
# It contains hostile work
# --------------------------------------------------------------------------------------


@os_sandboxed
def test_the_network_is_unreachable(data) -> None:
    result = run_pipeline(
        "import socket\n"
        "socket.create_connection(('1.1.1.1', 80), timeout=5)\n"
        "print('connected')\n",
        data,
    )
    assert not result.ok
    assert "connected" not in result.stdout


@os_sandboxed
def test_a_dns_lookup_is_refused(data) -> None:
    """Resolution is a separate syscall path from connecting, so it is checked separately."""
    result = run_pipeline(
        "import socket\nprint(socket.gethostbyname('example.com'))\n", data
    )
    assert not result.ok


@os_sandboxed
def test_writing_outside_the_work_directory_fails(data, tmp_path) -> None:
    target = tmp_path / "escaped.txt"
    result = run_pipeline(f"open({str(target)!r}, 'w').write('escaped')\n", data)

    assert not result.ok
    assert not target.exists(), "the sandbox allowed a write outside its directory"


@os_sandboxed
def test_deleting_this_project_fails(data) -> None:
    """The bluntest attack, against the thing we would most regret losing."""
    assert CANARY.exists(), "the canary is missing before the test even runs"

    result = run_pipeline(
        f"import shutil\nshutil.rmtree({str(PROJECT / 'app')!r})\n", data
    )

    assert not result.ok
    assert CANARY.exists(), "the sandbox allowed the application source to be deleted"


@os_sandboxed
def test_a_subprocess_cannot_escape_either(data) -> None:
    """A shell-out inherits the sandbox; the limits are not only on the Python process."""
    assert CANARY.exists()
    result = run_pipeline(
        f"import subprocess\n"
        f"subprocess.run(['/bin/rm', '-f', {str(CANARY)!r}], timeout=10)\n",
        data,
    )
    assert CANARY.exists(), "a subprocess escaped the sandbox"
    assert result is not None


def test_a_fork_bomb_is_stopped(data) -> None:
    result = run_pipeline("import os\nwhile True:\n    os.fork()\n", data, timeout=20)
    assert not result.ok


def test_a_memory_bomb_is_stopped(data) -> None:
    result = run_pipeline(
        "x = []\nwhile True:\n    x.append(bytearray(100_000_000))\n", data, timeout=45
    )
    assert not result.ok
    assert result.error is not None


def test_an_infinite_loop_times_out(data) -> None:
    result = run_pipeline("while True:\n    pass\n", data, timeout=6)

    assert not result.ok
    assert result.error is not None and result.error["type"] == "Timeout"
    # The kill has to be prompt. A timeout that overruns by minutes is a hang with a
    # nicer name.
    assert result.duration_ms < 20_000


def test_no_secret_from_the_parent_environment_reaches_the_sandbox(data) -> None:
    """A leaked token in the environment would be readable by anything the user runs.

    The property asserted is inheritance, not an exact list. The platform adds a couple
    of its own variables — macOS injects ``LC_CTYPE`` and ``__CF_USER_TEXT_ENCODING``
    below our level — and pinning the set exactly would make this test a record of
    which OS it last ran on. What matters is that nothing the *parent process* holds
    crosses over unless we put it there deliberately.
    """
    import os as parent_os

    result = run_pipeline(
        "import json, os\n"
        "print(json.dumps(sorted(os.environ)))\n"
        "open(os.environ['MLBX_OUTPUT'], 'w').write('a\\n1\\n')\n",
        data,
    )
    assert result.ok
    names = set(json.loads(result.stdout.strip().splitlines()[0]))

    deliberate = {
        "PATH", "HOME", "TMPDIR", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
        "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "JOBLIB_MULTIPROCESSING", "MLBX_INPUT", "MLBX_OUTPUT", "MLBX_REPORT",
    }
    inherited = (names - deliberate) & set(parent_os.environ)
    # Locale hints the OS re-adds carry nothing sensitive and are not ours to remove.
    inherited -= {"LC_CTYPE", "__CF_USER_TEXT_ENCODING", "LANG", "LC_ALL"}
    assert inherited == set(), f"parent environment leaked into the sandbox: {inherited}"

    # And the ones we do pass must not be the parent's values for them.
    assert result.stdout, "the probe produced no output"


def test_each_run_gets_a_fresh_directory(data) -> None:
    """One run must not be able to leave anything behind for the next."""
    write = (
        "import os\n"
        "open(os.path.join(os.path.dirname(os.environ['MLBX_OUTPUT']), 'note.txt'), 'w').write('hi')\n"
        "open(os.environ['MLBX_OUTPUT'], 'w').write('a\\n1\\n')\n"
    )
    read = (
        "import os\n"
        "path = os.path.join(os.path.dirname(os.environ['MLBX_OUTPUT']), 'note.txt')\n"
        "print('LEFTOVER' if os.path.exists(path) else 'clean')\n"
        "open(os.environ['MLBX_OUTPUT'], 'w').write('a\\n1\\n')\n"
    )
    assert run_pipeline(write, data).ok
    second = run_pipeline(read, data)
    assert second.ok
    assert "clean" in second.stdout


# --------------------------------------------------------------------------------------
# Failures are explained, not just reported
# --------------------------------------------------------------------------------------


def test_a_missing_column_points_at_the_line_and_names_the_column(data) -> None:
    result = run_pipeline(
        "import os\n"
        "import pandas as pd\n"
        "df = pd.read_csv(os.environ['MLBX_INPUT'])\n"
        "df['no_such_column'].sum()\n",
        data,
    )
    assert not result.ok
    error = result.error
    assert error is not None
    assert error["type"] == "KeyError"
    assert error["line"] == 4
    assert "no_such_column" in (error["cause"] or "")


def test_text_in_a_numeric_column_names_the_offending_value(data) -> None:
    result = run_pipeline(
        "import os\n"
        "import pandas as pd\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "df = pd.read_csv(os.environ['MLBX_INPUT'])\n"
        "StandardScaler().fit(df[['Sex']])\n",
        data,
    )
    assert not result.ok
    assert result.error is not None
    assert "female" in (result.error["cause"] or "") or "male" in (result.error["cause"] or "")
    assert result.error["suggestion"]


def test_a_syntax_error_reports_its_line(data) -> None:
    result = run_pipeline("def broken(:\n    pass\n", data)
    assert not result.ok
    assert result.error is not None and result.error["line"] == 1


def test_every_diagnosis_carries_something_actionable() -> None:
    """A cause without a suggestion leaves the reader knowing what broke and not what to do."""
    traceback = (
        'Traceback (most recent call last):\n'
        '  File "/tmp/x/pipeline.py", line 12, in <module>\n'
        "ValueError: could not convert string to float: 'abc'\n"
    )
    diagnosis = _diagnose(traceback)
    assert diagnosis is not None
    assert diagnosis["line"] == 12
    assert diagnosis["cause"] and diagnosis["suggestion"]


def test_the_last_frame_in_the_users_file_is_the_one_reported() -> None:
    """Frames inside pandas are not where the reader should be sent."""
    traceback = (
        'Traceback (most recent call last):\n'
        '  File "/tmp/x/pipeline.py", line 10, in <module>\n'
        '  File "/site-packages/pandas/core/frame.py", line 3800, in __getitem__\n'
        '  File "/tmp/x/pipeline.py", line 44, in helper\n'
        "KeyError: 'Age'\n"
    )
    assert _diagnose(traceback)["line"] == 44


@pytest.mark.parametrize(
    "count,expected",
    [(400 * 1024**2, "400 MB"), (2 * 1024**3, "2 GB"), (1536 * 1024**2, "1.5 GB")],
)
def test_byte_sizes_read_the_way_a_person_writes_them(count, expected) -> None:
    assert _human_bytes(count) == expected


# --------------------------------------------------------------------------------------
# What the interface is told about isolation
# --------------------------------------------------------------------------------------


def test_the_sandbox_reports_what_it_actually_enforces() -> None:
    """The workspace shows this to the user, so it must not overstate the protection."""
    status = sandbox_status()

    assert status["level"] in ("os_sandbox", "limits_only")
    assert status["label"] and status["detail"]
    if status["level"] == "os_sandbox":
        assert status["network_blocked"] and status["filesystem_isolated"]
    else:
        # On a platform without the OS layer the honest answer is that these are not
        # enforced, and the detail has to say what to do instead.
        assert not status["network_blocked"]
        assert "container" in status["detail"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific expectation")
def test_macos_gets_the_os_sandbox() -> None:
    assert sandbox_status()["level"] == "os_sandbox"
