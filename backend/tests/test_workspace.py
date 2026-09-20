"""The workspace API, end to end.

These drive the app the way the interface does: open a workspace, change a toggle, read
the impact, compile, look at the result, edit the code, break it, fix it, restore an
earlier state, download the artefacts.

The properties worth defending here are the ones the product's claims rest on:

* a toggle and the code cannot disagree, because the code is generated from the toggle;
* hand-edited code is never silently overwritten, and the app says when it is what runs;
* an impact warning quotes measured numbers, and says so when a step is doing nothing;
* a failed compile is kept, not discarded.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURE = Path(__file__).parent / "fixtures" / "titanic.csv"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def workspace(client) -> dict:
    with FIXTURE.open("rb") as handle:
        dataset = client.post(
            "/api/datasets", files={"file": ("titanic.csv", handle, "text/csv")}
        ).json()
    response = client.post(f"/api/workspaces/from-dataset/{dataset['id']}")
    assert response.status_code == 201, response.text
    return response.json()


def set_step(client, workspace: dict, step: str, **changes) -> dict:
    """Change one step's configuration and return the updated workspace."""
    config = {**workspace["config"], step: {**workspace["config"][step], **changes}}
    response = client.put(
        f"/api/workspaces/{workspace['id']}/config", json={"config": config, "summary": ""}
    )
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------------------
# Opening
# --------------------------------------------------------------------------------------


def test_opening_a_workspace_gives_everything_the_interface_needs(workspace) -> None:
    assert workspace["dataset"]["rows"] == 891
    assert workspace["config"]["target"] == "Survived"
    assert workspace["code"].startswith('"""Preprocessing pipeline')
    assert workspace["code_is_edited"] is False
    assert len(workspace["steps"]) == 8
    assert workspace["baseline_health"]["score"] > 0
    assert len(workspace["baseline_health"]["dimensions"]) == 6
    assert workspace["sandbox"]["level"] in ("os_sandbox", "limits_only")


def test_a_dataset_has_exactly_one_workspace(client, workspace) -> None:
    """Asking twice must not discard the work in the first one."""
    again = client.post(f"/api/workspaces/from-dataset/{workspace['dataset']['id']}")
    assert again.json()["id"] == workspace["id"]


def test_every_step_reports_what_it_is_doing_to_this_dataset(workspace) -> None:
    steps = {step["id"]: step for step in workspace["steps"]}

    # Real numbers from the fixture, not placeholders.
    assert "885" in steps["missing_values"]["summary"]
    assert steps["duplicates"]["summary"] == "No duplicate rows found."
    assert steps["duplicates"]["status"] == "idle"
    assert steps["missing_values"]["status"] == "active"
    assert steps["data_types"]["locked"] is True


def test_a_missing_workspace_is_a_404(client) -> None:
    assert client.get("/api/workspaces/nope").status_code == 404


# --------------------------------------------------------------------------------------
# Configuration drives the code
# --------------------------------------------------------------------------------------


def test_changing_a_toggle_regenerates_the_code(client, workspace) -> None:
    assert "StandardScaler()" in workspace["code"]

    updated = set_step(client, workspace, "scaling", method="robust")

    assert "RobustScaler()" in updated["code"]
    assert "StandardScaler()" not in updated["code"]
    assert updated["code_is_edited"] is False


def test_disabling_a_step_removes_it_from_the_code_entirely(client, workspace) -> None:
    """Off means absent. A commented-out block is clutter the reader has to dismiss."""
    updated = set_step(client, workspace, "scaling", enabled=False)

    assert "RobustScaler" not in updated["code"]
    assert "StandardScaler" not in updated["code"]
    assert "# StandardScaler" not in updated["code"]


def test_the_code_always_matches_the_configuration(client, workspace) -> None:
    """The invariant that stops the toggles and the editor drifting apart."""
    updated = set_step(client, workspace, "outliers", action="remove", method="zscore")
    assert updated["code"] == updated["generated_code"]


def test_an_invalid_option_is_refused(client, workspace) -> None:
    config = {**workspace["config"], "scaling": {**workspace["config"]["scaling"], "method": "quantile"}}
    response = client.put(
        f"/api/workspaces/{workspace['id']}/config", json={"config": config}
    )
    assert response.status_code == 422


def test_a_change_is_recorded_in_the_history(client, workspace) -> None:
    set_step(client, workspace, "scaling", method="minmax")
    history = client.get(f"/api/workspaces/{workspace['id']}/history").json()

    assert any("minmax" in entry["summary"] for entry in history)
    assert history[-1]["kind"] == "created"


# --------------------------------------------------------------------------------------
# Impact preview
# --------------------------------------------------------------------------------------


def test_the_impact_of_disabling_a_step_is_measured(client, workspace) -> None:
    response = client.post(
        f"/api/workspaces/{workspace['id']}/impact",
        json={"step": "missing_values", "enabled": False},
    )
    assert response.status_code == 200
    report = response.json()

    assert report["severity"] == "blocking"
    assert any("885" in line for line in report["consequences"])
    # The score is measured by running the pipeline both ways, so both must be present.
    assert isinstance(report["current_score"], int)
    assert isinstance(report["projected_score"], int)
    assert report["delta"] == report["projected_score"] - report["current_score"]


def test_a_step_doing_nothing_says_so_rather_than_inventing_a_warning(client, workspace) -> None:
    """The fixture has no duplicates. Claiming otherwise would be the easy, wrong move."""
    report = client.post(
        f"/api/workspaces/{workspace['id']}/impact",
        json={"step": "duplicates", "enabled": False},
    ).json()

    assert report["severity"] == "none"
    assert "nothing" in report["consequences"][0].lower()


def test_every_step_produces_a_usable_warning(client, workspace) -> None:
    for step in workspace["step_order"]:
        report = client.post(
            f"/api/workspaces/{workspace['id']}/impact", json={"step": step, "enabled": False}
        ).json()
        assert report["consequences"], f"{step} produced no consequences"
        assert report["severity"] in ("none", "low", "moderate", "high", "blocking")
        assert report["label"]


def test_an_unknown_step_is_refused(client, workspace) -> None:
    response = client.post(
        f"/api/workspaces/{workspace['id']}/impact",
        json={"step": "not_a_step", "enabled": False},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------------------
# Compiling
# --------------------------------------------------------------------------------------


def test_compiling_runs_the_pipeline_and_reports_the_change(client, workspace) -> None:
    response = client.post(f"/api/workspaces/{workspace['id']}/compile", json={"source": "auto"})
    assert response.status_code == 200
    compilation = response.json()["compilation"]

    assert compilation["status"] == "success", compilation.get("error")
    assert compilation["source"] == "pipeline"
    assert compilation["duration_ms"] > 0
    assert compilation["logs"]

    result = compilation["result"]
    assert result["rows_before"] == 891
    assert result["columns_after"] != result["columns_before"]
    assert result["steps"], "the per-step report is missing"
    assert result["comparison"]["score"]["before"] >= 0


def test_the_pipeline_improves_health_on_the_fixture(client, workspace) -> None:
    """The product's central claim, checked against a real dataset.

    Not a tautology: the health engine and the pipeline are independent, and an earlier
    version of this pipeline made the score *worse* by emitting constant columns.
    """
    compilation = client.post(
        f"/api/workspaces/{workspace['id']}/compile", json={"source": "auto"}
    ).json()["compilation"]

    before = compilation["health_before"]["score"]
    after = compilation["health_after"]["score"]
    assert after >= before, f"preprocessing lowered health from {before} to {after}"


def test_compiling_fills_the_missing_values(client, workspace) -> None:
    client.post(f"/api/workspaces/{workspace['id']}/compile", json={"source": "auto"})
    page = client.get(
        f"/api/workspaces/{workspace['id']}/preview",
        params={"which": "processed", "page_size": 200},
    ).json()

    blanks = sum(column["missing"] for column in page["columns"])
    assert blanks == 0


def test_a_failing_compile_is_kept_and_explained(client, workspace) -> None:
    client.put(
        f"/api/workspaces/{workspace['id']}/code",
        json={"code": "import pandas as pd\nraise ValueError('deliberate')\n"},
    )
    response = client.post(f"/api/workspaces/{workspace['id']}/compile", json={"source": "auto"})
    compilation = response.json()["compilation"]

    assert compilation["status"] == "error"
    assert compilation["error"]["type"] == "ValueError"
    assert compilation["error"]["line"] == 2

    # Kept, not discarded: the user needs to come back to it.
    history = client.get(f"/api/workspaces/{workspace['id']}/compilations").json()
    assert history[0]["status"] == "error"


def test_compiling_twice_keeps_both_runs(client, workspace) -> None:
    client.post(f"/api/workspaces/{workspace['id']}/compile", json={"source": "auto"})
    set_step(client, workspace, "scaling", method="robust")
    client.post(f"/api/workspaces/{workspace['id']}/compile", json={"source": "auto"})

    assert len(client.get(f"/api/workspaces/{workspace['id']}/compilations").json()) == 2


# --------------------------------------------------------------------------------------
# The editor
# --------------------------------------------------------------------------------------


def test_editing_the_code_makes_it_authoritative(client, workspace) -> None:
    edited = workspace["code"] + "\n# a line only I would write\n"
    updated = client.put(
        f"/api/workspaces/{workspace['id']}/code", json={"code": edited}
    ).json()

    assert updated["code_is_edited"] is True
    assert updated["code_is_authoritative"] is True


def test_a_toggle_never_discards_hand_edited_code(client, workspace) -> None:
    """The worst thing this app could do, so it is checked explicitly."""
    edited = workspace["code"] + "\n# a note I do not want to lose\n"
    client.put(f"/api/workspaces/{workspace['id']}/code", json={"code": edited})

    after_toggle = set_step(client, workspace, "scaling", method="minmax")

    assert "a note I do not want to lose" in after_toggle["code"]
    assert after_toggle["code_is_edited"] is True
    # The generated version is still available, and the app admits the two differ.
    assert "MinMaxScaler" in after_toggle["generated_code"]
    assert after_toggle["code_is_authoritative"] is True


def test_typing_back_to_the_generated_text_is_not_an_edit(client, workspace) -> None:
    """Otherwise a workspace stays flagged as diverged after an undo."""
    client.put(
        f"/api/workspaces/{workspace['id']}/code", json={"code": workspace["code"] + "\n# x"}
    )
    restored = client.put(
        f"/api/workspaces/{workspace['id']}/code", json={"code": workspace["code"]}
    ).json()

    assert restored["code_is_edited"] is False


def test_resetting_discards_the_edits(client, workspace) -> None:
    client.put(f"/api/workspaces/{workspace['id']}/code", json={"code": "print('mine')"})
    reset = client.post(f"/api/workspaces/{workspace['id']}/code/reset").json()

    assert reset["code_is_edited"] is False
    assert reset["code"] == reset["generated_code"]
    assert "print('mine')" not in reset["code"]


def test_edited_code_is_what_actually_runs(client, workspace) -> None:
    marker = "EDITED-CODE-RAN"
    edited = workspace["code"].replace(
        'record("load"', f'print("{marker}")\nrecord("load"', 1
    )
    client.put(f"/api/workspaces/{workspace['id']}/code", json={"code": edited})

    compilation = client.post(
        f"/api/workspaces/{workspace['id']}/compile", json={"source": "auto"}
    ).json()["compilation"]

    assert compilation["source"] == "editor"
    assert any(marker in line["text"] for line in compilation["logs"])


# --------------------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------------------


def test_the_preview_marks_blanks_and_outliers(client, workspace) -> None:
    page = client.get(
        f"/api/workspaces/{workspace['id']}/preview", params={"page_size": 200}
    ).json()

    assert page["total_rows"] == 891
    cells = [cell for row in page["rows"] for cell in row["cells"].values()]
    assert any(cell["missing"] for cell in cells), "no blanks marked"
    assert any(cell["outlier"] for cell in cells), "no outliers marked"


def test_the_preview_paginates_sorts_and_searches(client, workspace) -> None:
    first = client.get(
        f"/api/workspaces/{workspace['id']}/preview",
        params={"page_size": 10, "sort": "Fare", "descending": True},
    ).json()
    assert len(first["rows"]) == 10

    fares = [row["cells"]["Fare"]["value"] for row in first["rows"]]
    assert fares == sorted(fares, reverse=True)

    searched = client.get(
        f"/api/workspaces/{workspace['id']}/preview", params={"search": "female"}
    ).json()
    assert 0 < searched["total_rows"] < 891


def test_asking_for_processed_rows_before_compiling_says_so(client, workspace) -> None:
    response = client.get(
        f"/api/workspaces/{workspace['id']}/preview", params={"which": "processed"}
    )
    assert response.status_code == 404
    assert "compiled" in response.json()["error"]


# --------------------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------------------


def test_an_earlier_state_can_be_restored(client, workspace) -> None:
    set_step(client, workspace, "scaling", method="minmax")
    changed = set_step(client, workspace, "scaling", method="robust")
    assert changed["config"]["scaling"]["method"] == "robust"

    history = client.get(f"/api/workspaces/{workspace['id']}/history").json()
    minmax_entry = next(e for e in history if "minmax" in e["summary"])

    restored = client.post(
        f"/api/workspaces/{workspace['id']}/history/{minmax_entry['id']}/restore"
    ).json()
    assert restored["config"]["scaling"]["method"] == "minmax"
    assert "MinMaxScaler" in restored["code"]


def test_restoring_is_itself_recorded(client, workspace) -> None:
    """Going back is part of what happened; erasing it would make the panel lie."""
    set_step(client, workspace, "scaling", method="minmax")
    history = client.get(f"/api/workspaces/{workspace['id']}/history").json()
    client.post(f"/api/workspaces/{workspace['id']}/history/{history[-1]['id']}/restore")

    after = client.get(f"/api/workspaces/{workspace['id']}/history").json()
    assert after[0]["kind"] == "restore"
    assert len(after) > len(history)


# --------------------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------------------


def test_code_and_config_download_without_compiling(client, workspace) -> None:
    code = client.get(f"/api/workspaces/{workspace['id']}/download/code")
    assert code.status_code == 200
    compile(code.text, "pipeline.py", "exec")

    config = client.get(f"/api/workspaces/{workspace['id']}/download/config")
    assert json.loads(config.text)["target"] == "Survived"


def test_data_downloads_require_a_successful_compile(client, workspace) -> None:
    assert client.get(f"/api/workspaces/{workspace['id']}/download/processed").status_code == 404

    client.post(f"/api/workspaces/{workspace['id']}/compile", json={"source": "auto"})

    processed = client.get(f"/api/workspaces/{workspace['id']}/download/processed")
    assert processed.status_code == 200
    assert len(processed.text.splitlines()) == 892  # header plus every row


def test_the_report_records_what_was_deliberately_not_done(client, workspace) -> None:
    """The section most reports omit, and the one that makes the rest interpretable."""
    set_step(client, workspace, "outliers", enabled=False)
    client.post(f"/api/workspaces/{workspace['id']}/compile", json={"source": "auto"})

    report = client.get(f"/api/workspaces/{workspace['id']}/download/report").text

    assert "## Steps deliberately not applied" in report
    assert "Outlier Treatment" in report.split("## Steps deliberately not applied")[1]
    assert "## The code that ran" in report
    assert "Completeness" in report


def test_the_project_archive_contains_everything(client, workspace) -> None:
    client.post(f"/api/workspaces/{workspace['id']}/compile", json={"source": "auto"})
    response = client.get(f"/api/workspaces/{workspace['id']}/download/project")

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "original.csv", "processed.csv", "pipeline.py",
            "pipeline_config.json", "report.md",
        }
        compile(archive.read("pipeline.py").decode(), "pipeline.py", "exec")


def test_an_unknown_download_is_refused(client, workspace) -> None:
    assert client.get(f"/api/workspaces/{workspace['id']}/download/secrets").status_code == 422
