import copy
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from analyst.cli import main
from analyst.reporting import dataset_provenance, render_report, report_payload


@pytest.fixture
def run():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "examples" / "report_run.json").read_text())


def test_export_preserves_metrics_warnings_and_does_not_mutate_source(run):
    original = copy.deepcopy(run)
    payload = report_payload(run)

    assert run == original
    assert payload["dataset"]["content_hash"] == run["dataset_provenance"]["content_hash"]
    assert payload["predictive_evidence"][0]["metrics"] == run["predictive_evidence"][0]["metrics"]
    assert payload["statistical_evidence"][0]["test"] == run["statistical_evidence"][0]["test"]
    assert payload["predictive_evidence"][1]["warnings"]
    assert payload["claims"] == []  # an unreviewed demo must never imply validation


def test_export_allowlist_excludes_internal_data_at_nested_levels(run):
    secret = "UNEXPORTED_INTERNAL_VALUE"
    run.update(options={"api_key": secret}, events=[{"stdout": secret}],
               data_dir=f"/private/{secret}", artifacts={"code": secret})
    run["dataset_provenance"]["root"] = secret
    run["dataset_provenance"]["tables"][0]["path"] = secret
    run["predictive_evidence"][0]["metrics"][0]["internal"] = secret
    run["predictive_evidence"][0]["diagnostics"]["stderr"] = secret
    run["data_quality_report"]["tables"] = [{"sample_rows": secret}]

    payload = report_payload(run)

    assert secret not in json.dumps(payload)
    assert secret not in render_report(payload)


@pytest.mark.parametrize("status", ["queued", "running", "error", None])
def test_incomplete_or_failed_runs_cannot_be_exported(run, status):
    run["status"] = status
    with pytest.raises(ValueError, match="completed runs"):
        report_payload(run)


def test_missing_answer_is_not_presented_as_a_completed_report(run):
    run["answer"] = "   "
    with pytest.raises(ValueError, match="no answer"):
        report_payload(run)


def test_html_escapes_untrusted_content_and_contains_no_active_resources(run):
    attack = '<script>alert("x")</script><img src="https://example.invalid/leak">'
    run["answer"] = attack
    run["question"] = attack
    run["predictive_evidence"][0]["warnings"] = [attack]
    run["council_findings"] = [{"specialist": "metric", "verdict": "challenge",
                                "conclusion": attack, "risks": [attack]}]
    markup = render_report(report_payload(run))

    class Resources(HTMLParser):
        def __init__(self):
            super().__init__()
            self.unsafe = []

        def handle_starttag(self, tag, attrs):
            if tag in {"script", "img", "iframe", "object", "link", "form"}:
                self.unsafe.append(tag)
            for name, value in attrs:
                if name.startswith("on") or (name in {"src", "href"} and not value.startswith("#")):
                    self.unsafe.append((name, value))

    parser = Resources()
    parser.feed(markup)
    assert parser.unsafe == []
    assert "&lt;script&gt;" in markup
    assert "Content-Security-Policy" in markup
    assert "default-src &#39;none&#39;" in markup


def test_report_makes_claim_caveats_failed_tasks_and_invalid_evidence_visible(run):
    run["claims"] = [{"task_id": "t1", "claim": "a claim", "verdict": "caveat",
                      "note": "Unresolved selection bias"}]
    run["dag"]["failed"] = {"id": "failed", "status": "failed", "goal": "Failed analysis"}
    run["predictive_evidence"].append({"status": "invalid", "request_id": "bad_model",
                                       "warnings": ["target cannot also be a feature"]})
    markup = render_report(report_payload(run))

    assert "1 claim caveat(s)/failure(s), 1 failed task(s), 1 invalid evidence request(s)" in markup
    assert "Unresolved selection bias" in markup
    assert "target cannot also be a feature" in markup


def test_old_completed_run_does_not_invent_missing_provenance():
    payload = report_payload({"status": "done", "answer": "A historical answer"})

    assert payload["dataset"] == {"content_hash": None, "tables": []}
    assert "Not recorded" in render_report(payload)
    assert "None recorded" in render_report(payload)


def test_portfolio_example_matches_the_current_renderer(run):
    root = Path(__file__).resolve().parents[1]
    assert (root / "examples" / "analysis-report.html").read_text() == render_report(report_payload(run))


def test_cli_graph_state_and_api_record_export_equivalently(run):
    cli = copy.deepcopy(run)
    cli["dataset"] = cli.pop("dataset_provenance")
    cli["statistical_evidence"] = {str(i): e for i, e in enumerate(cli["statistical_evidence"])}
    cli["predictive_evidence"] = {str(i): e for i, e in enumerate(cli["predictive_evidence"])}

    assert report_payload(cli) == report_payload(run)
    assert dataset_provenance(cli["dataset"]) == run["dataset_provenance"]


@pytest.mark.parametrize("suffix", ["html", "json"])
def test_cli_exports_both_formats_without_provider_access(tmp_path, run, suffix, monkeypatch):
    monkeypatch.setattr("analyst.llm.client.make_client", lambda *a, **kw: pytest.fail("LLM access"))
    source = tmp_path / "run.json"
    source.write_text(json.dumps(run))
    output = tmp_path / f"report.{suffix}"

    assert main(["export-report", str(source), "--out", str(output)]) == 0
    assert ("<!doctype html>" in output.read_text() if suffix == "html"
            else json.loads(output.read_text())["report_version"] == "analysis-report-v1")


def test_cli_does_not_replace_existing_output_on_invalid_input(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text('{"status": "running"}')
    output = tmp_path / "report.html"
    output.write_text("previous export")

    assert main(["export-report", str(source), "--out", str(output)]) == 1
    assert output.read_text() == "previous export"
