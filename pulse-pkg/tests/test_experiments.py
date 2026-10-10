import json
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pulse import pulse_experiments as experiments
from pulse.pulse_cli import PulseCLI


RUNNER = """
import json, pathlib, sys
config = json.loads(pathlib.Path(sys.argv[1]).read_text())
state = pathlib.Path("intervention.txt").read_text().strip()
proxy_failed = config["hidden_size"] >= 12 and state == "unchanged"
for step in range(3):
    loss = (100.0 + step) if proxy_failed else (1.0 / (step + 1))
    print("PULSE_METRICS: " + json.dumps({"loss": loss, "grad_norm": 2.0, "step": step}))
if state == "candidate-fails":
    raise SystemExit(1)
"""


def spec_for(project, scales=(0.25, 0.5, 0.75), candidates=None):
    (project / "config.json").write_text(json.dumps({"hidden_size": 16, "num_attention_heads": 4}))
    (project / "intervention.txt").write_text("unchanged")
    return {
        "command": [sys.executable, "-c", RUNNER, "{config}"],
        "proxy": {"config_file": "config.json", "scales": list(scales)},
        "failure_signature": {
            "kind": "metric_threshold", "metric": "loss", "operator": "gt", "threshold": 50,
        },
        "original_metrics": [1.0, 100.0],
        "original_evidence": "The live run's loss reached 100 at step 42.",
        "repetitions": 2,
        "max_parallel": 2,
        "objective": {"metric": "loss", "direction": "min"},
        "validation_scales": [1.0],
        "candidates": candidates or [{
            "name": "stabilize-loss",
            "hypothesis": "The intervention removes the observed divergence.",
            "replacements": [{
                "file": "intervention.txt", "find": "unchanged", "replace": "fixed",
            }],
        }],
    }


def test_proxy_gate_retries_larger_scales_then_validates_isolated_candidate(tmp_path):
    spec = spec_for(tmp_path)

    report = experiments.run_experiment(tmp_path, spec)

    assert report["status"] == "completed"
    assert [attempt["scale"] for attempt in report["proxy_attempts"]] == [0.25, 0.5, 0.75]
    assert report["reproduction"]["passed"]
    assert report["reproduction"]["confidence"] >= 0.8
    assert report["reproduction"]["scale"] == 0.75
    assert {branch["name"] for branch in report["branches"]} == {"control", "stabilize-loss"}
    candidate = next(branch for branch in report["branches"] if branch["name"] == "stabilize-loss")
    assert candidate["successful_trials"] == 2
    assert candidate["validated"] is True
    assert report["validation"][0]["scope"] == "isolated source-scale copy"
    assert candidate["metric_summary"]["loss"]["first"] == 1.0
    assert candidate["metric_summary"]["loss"]["last"] == 0.3333333333333333
    assert report["proxy_attempts"][0]["proxy_dimensions"]["hidden_size"] == 4
    assert (tmp_path / "config.json").read_text() == json.dumps(
        {"hidden_size": 16, "num_attention_heads": 4}
    )
    assert (tmp_path / "intervention.txt").read_text() == "unchanged"


def test_candidates_are_not_run_when_no_proxy_reproduces_failure(tmp_path):
    spec = spec_for(tmp_path, scales=(0.25, 0.5))

    report = experiments.run_experiment(tmp_path, spec)

    assert report["status"] == "inconclusive"
    assert report["stage"] == "reproduction_gate"
    assert not report["reproduction"].get("passed", False)
    assert report["branches"] == []
    assert report["run_count"] == 4
    assert (tmp_path / "intervention.txt").read_text() == "unchanged"


def test_reproduction_gate_scans_early_metrics_beyond_output_tail(tmp_path):
    spec = spec_for(tmp_path, scales=(0.75,))
    spec["command"] = [
        sys.executable, "-c",
        "import json,pathlib,sys; "
        "failed=pathlib.Path('intervention.txt').read_text().strip()=='unchanged'; "
        "print('PULSE_METRICS: '+json.dumps({'loss':100.0 if failed else 1.0}),flush=True); "
        "sys.stdout.write('filler'*20000); sys.stdout.flush(); "
        "print(); print('PULSE_METRICS: '+json.dumps({'loss':1.0}),flush=True)",
        "{config}",
    ]

    report = experiments.run_experiment(tmp_path, spec)

    assert report["reproduction"]["passed"]
    assert report["proxy_attempts"][0]["reproduced"] == 2
    assert report["proxy_attempts"][0]["metrics"][0]["loss"] == 1.0


def test_stderr_signature_scans_entire_output_not_only_tail():
    outcome = experiments._run_command(
        [
            sys.executable, "-c",
            "import sys; sys.stderr.write('TARGET FAILURE\\\\n'+'filler'*20000)",
        ],
        Path.cwd(), {}, 10, threading.Event(),
        {"kind": "stderr_regex", "pattern": "target failure"},
    )

    assert outcome["failure_observed"]
    assert outcome["evidence_complete"]


def test_nonzero_candidate_run_is_not_scored_as_a_fix(tmp_path):
    candidate = {
        "name": "broken-runner",
        "hypothesis": "This candidate's runner exits unsuccessfully.",
        "replacements": [{
            "file": "intervention.txt", "find": "unchanged", "replace": "candidate-fails",
        }],
    }
    report = experiments.run_experiment(tmp_path, spec_for(tmp_path, candidates=[candidate]))

    branch = next(item for item in report["branches"] if item["name"] == "broken-runner")
    assert branch["successful_trials"] == 0
    assert branch["success_rate"] == 0


def test_adaptive_budget_adds_trials_to_promising_candidate_and_control(tmp_path):
    spec = spec_for(tmp_path)
    spec["adaptive_extra_repetitions"] = 1
    spec["adaptive_top"] = 1

    report = experiments.run_experiment(tmp_path, spec)

    assert {item["branch"] for item in report["adaptive_allocations"]} == {
        "control", "stabilize-loss",
    }
    assert all(branch["trials"] == 3 for branch in report["branches"])


def test_threshold_must_match_original_evidence_before_any_command_runs(tmp_path):
    spec = spec_for(tmp_path)
    spec["original_metrics"] = [1.0, 2.0]

    with pytest.raises(experiments.ExperimentError, match="do not meet"):
        experiments.run_experiment(tmp_path, spec)


def test_nonfinite_gate_requires_nonfinite_original_evidence(tmp_path):
    spec = spec_for(tmp_path)
    spec["failure_signature"] = {"kind": "metric_nonfinite", "metric": "loss"}

    with pytest.raises(experiments.ExperimentError, match="non-finite original_metrics"):
        experiments.run_experiment(tmp_path, spec)


def test_spec_and_patch_paths_cannot_escape_project(tmp_path):
    spec = spec_for(tmp_path)
    spec["proxy"]["config_file"] = "../outside.json"

    with pytest.raises(experiments.ExperimentError, match="stay inside"):
        experiments.run_experiment(tmp_path, spec)


def test_gpu_resource_class_requires_explicit_device(tmp_path):
    spec = spec_for(tmp_path)
    spec["resource_class"] = "gpu"

    with pytest.raises(experiments.ExperimentError, match="requires explicit devices"):
        experiments.run_experiment(tmp_path, spec)


def test_candidate_targets_are_validated_before_baseline_execution(tmp_path):
    spec = spec_for(tmp_path)
    spec["candidates"][0]["replacements"][0]["file"] = "../outside.txt"

    with pytest.raises(experiments.ExperimentError, match="stay inside"):
        experiments.run_experiment(tmp_path, spec)


def test_experiment_directive_is_parsed_and_serviced():
    cleaned, requests = PulseCLI._extract_new_directives(
        "Hypotheses are prepared.\nEXPERIMENT: experiments/failure.json"
    )

    assert "EXPERIMENT:" not in cleaned
    assert requests["experiment"] == ["experiments/failure.json"]


def test_cli_experiment_job_can_be_inspected_and_cancelled(tmp_path, monkeypatch):
    from pulse import pulse_cli

    spec = spec_for(tmp_path, scales=(0.75,))
    spec["command"] = [sys.executable, "-c", "import time; time.sleep(30)", "{config}"]
    (tmp_path / "experiment.json").write_text(json.dumps(spec))
    cli = PulseCLI.__new__(PulseCLI)
    cli._project_root = str(tmp_path)
    cli.scalar_histories = {}
    cli._log_incident = lambda *args, **kwargs: None
    monkeypatch.setattr(pulse_cli._ui, "message", lambda _message: None)

    started = cli._run_experiment("experiment.json")
    job_id = started.split()[1]
    assert "started" in started
    assert "running" in cli._run_experiment(f"status {job_id}")
    assert "cancellation requested" in cli._run_experiment(f"cancel {job_id}")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = cli._run_experiment(f"status {job_id}")
        if "cancelled" in status:
            break
        time.sleep(0.05)
    assert "cancelled" in status
