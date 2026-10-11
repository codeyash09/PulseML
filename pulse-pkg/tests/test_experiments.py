import json
import os
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
        if status.startswith(f"EXPERIMENT {job_id}: cancelled"):
            break
        time.sleep(0.05)
    # Every report mentions "cancelled" (each run has a cancelled flag): check the state.
    assert status.splitlines()[0] == f"EXPERIMENT {job_id}: cancelled"


def bare_cli(project, monkeypatch):
    from pulse import pulse_cli

    cli = PulseCLI.__new__(PulseCLI)
    cli._project_root = str(project)
    cli.scalar_histories = {}
    cli._log_incident = lambda *args, **kwargs: None
    monkeypatch.setattr(pulse_cli._ui, "message", lambda _message: None)
    return cli


def wait_for_state(cli, job_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = cli._run_experiment(f"status {job_id}").splitlines()[0]
        if not state.endswith(": running"):
            return state
        time.sleep(0.05)
    raise AssertionError("experiment did not finish")


def test_cli_flagged_experiment_command_is_confirmed_and_a_decline_starts_nothing(tmp_path, monkeypatch):
    spec = spec_for(tmp_path, scales=(0.75,))
    spec["command"] = ["rm", "-rf", "{project}/scratch"]
    (tmp_path / "experiment.json").write_text(json.dumps(spec))
    cli = bare_cli(tmp_path, monkeypatch)
    asked = []

    def decline(command, flags, purpose=None):
        asked.append(command)
        return False

    cli._confirm_terminal_command = decline
    result = cli._run_experiment("experiment.json")

    assert asked == ["rm -rf ./scratch"]
    assert "not started" in result and "declined" in result
    assert cli._experiment_jobs == {}


def test_cli_confirms_each_distinct_experiment_command_once(tmp_path, monkeypatch):
    spec = spec_for(tmp_path, scales=(0.75,))
    spec["command"] = [sys.executable, "-c", "pass", "{seed}", "{config}"]
    (tmp_path / "experiment.json").write_text(json.dumps(spec))
    cli = bare_cli(tmp_path, monkeypatch)
    cli._terminal_needs_confirmation = lambda command: {"deletes_files": True}
    asked = []
    cli._confirm_terminal_command = lambda command, flags, purpose=None: asked.append(command) or True

    result = cli._run_experiment("experiment.json")
    job_id = result.split()[1]
    wait_for_state(cli, job_id)

    assert "started" in result
    assert len(asked) == len(set(asked)) == 2          # seeds 0 and 1; {branch} is not used
    assert all(command.endswith(" config.json") for command in asked)


def test_cli_invalid_spec_is_reported_before_any_job_starts(tmp_path, monkeypatch):
    spec = spec_for(tmp_path)
    spec["repetitions"] = 0
    (tmp_path / "experiment.json").write_text(json.dumps(spec))
    cli = bare_cli(tmp_path, monkeypatch)

    result = cli._run_experiment("experiment.json")

    assert "not started -- invalid spec" in result and "repetitions" in result
    assert cli._experiment_jobs == {}
    assert "not started" in cli._run_experiment("missing.json")


def test_cli_status_is_a_compact_summary(tmp_path, monkeypatch):
    (tmp_path / "experiment.json").write_text(json.dumps(spec_for(tmp_path, scales=(0.75,))))
    cli = bare_cli(tmp_path, monkeypatch)

    job_id = cli._run_experiment("experiment.json").split()[1]
    assert wait_for_state(cli, job_id) == f"EXPERIMENT {job_id}: completed"
    status = cli._run_experiment(f"status {job_id}")

    assert "conclusion:" in status and "branch stabilize-loss" in status
    assert "metric_summary" not in status and len(status) < 2000


def test_candidate_must_pass_every_validation_scale(tmp_path):
    spec = spec_for(tmp_path, scales=(0.75,))
    spec["validation_scales"] = [0.5, 1.0]
    # The fix holds at the proxy and at full scale but breaks at scale 0.5 (hidden_size 8).
    spec["command"] = [
        sys.executable, "-c",
        "import json,pathlib,sys; c=json.loads(pathlib.Path(sys.argv[1]).read_text()); "
        "s=pathlib.Path('intervention.txt').read_text().strip(); "
        "bad=(s=='unchanged' and c['hidden_size']>=12) or (s=='fixed' and c['hidden_size']==8); "
        "print('PULSE_METRICS: '+json.dumps({'loss':100.0 if bad else 1.0}))",
        "{config}",
    ]

    report = experiments.run_experiment(tmp_path, spec)

    assert [(item["scale"], item["passed"]) for item in report["validation"]] == [(0.5, False), (1.0, True)]
    candidate = next(branch for branch in report["branches"] if branch["name"] == "stabilize-loss")
    assert candidate["validated"] is False


def test_secret_like_metric_names_do_not_break_the_report(tmp_path):
    spec = spec_for(tmp_path, scales=(0.75,))
    spec["command"] = [
        sys.executable, "-c",
        "import json; print('PULSE_METRICS: '+json.dumps({'loss':100.0,'password':0.123456789,"
        "'token':12345678.5,'note':'api_key=abcdefgh12345678'}))",
    ]

    report = experiments.run_experiment(tmp_path, spec)

    metrics = report["runs"][0]["metrics"]
    assert metrics["password"] == 0.123456789 and metrics["token"] == 12345678.5
    assert "abcdefgh12345678" not in json.dumps(report)


def test_gpu_devices_are_never_shared_by_concurrent_runs(tmp_path):
    locks = tmp_path.parent / (tmp_path.name + "-locks")
    locks.mkdir()
    spec = spec_for(tmp_path, scales=(0.75,))
    # Equal seeds: a seed-hash assignment would put every run on the same device.
    spec.update({"resource_class": "gpu", "devices": ["a", "b"], "max_parallel": 2, "repetitions": 6,
                 "seeds": [0] * 6,
                 "failure_signature": {"kind": "stderr_regex", "pattern": "DEVICE CLASH"}})
    spec["command"] = [
        sys.executable, "-c",
        "import os,sys,time; p=os.path.join(sys.argv[1], os.environ['CUDA_VISIBLE_DEVICES'])\n"
        "try: fd=os.open(p, os.O_CREAT|os.O_EXCL)\n"
        "except FileExistsError: sys.stderr.write('DEVICE CLASH'); raise SystemExit(1)\n"
        "time.sleep(0.3); os.close(fd); os.remove(p)",
        str(locks),
    ]

    report = experiments.run_experiment(tmp_path, spec)

    assert report["proxy_attempts"][0]["reproduced"] == 0
    assert {run["device"] for run in report["runs"]} == {"a", "b"}
    assert all(run["returncode"] == 0 for run in report["runs"])


def test_timed_out_run_is_not_a_reproduced_nonzero_exit():
    outcome = experiments._run_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        Path.cwd(), dict(os.environ), 0.3, threading.Event(), {"kind": "exit_code_nonzero"},
    )

    assert outcome["timed_out"]
    assert not outcome["failure_observed"]


@pytest.mark.skipif(os.name == "nt", reason="process groups")
def test_runner_grandchildren_are_killed_with_the_group(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    experiments._run_command(
        [sys.executable, "-c",
         "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
         f"open({str(pid_file)!r},'w').write(str(p.pid))"],
        tmp_path, dict(os.environ), 10, threading.Event(), {"kind": "exit_code_nonzero"},
    )
    pid = int(pid_file.read_text())

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        except OSError:
            break                   # gone
        if state == "Z":
            break                   # killed, waiting for its new parent to reap it
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        raise AssertionError("the runner's grandchild survived")


def test_cancel_all_stops_running_and_queued_runs(tmp_path):
    spec = spec_for(tmp_path, scales=(0.75,))
    spec.update({"resource_class": "cpu", "max_parallel": 1, "repetitions": 4})
    spec["command"] = [sys.executable, "-c", "import time; time.sleep(60)"]
    reports = []
    thread = threading.Thread(target=lambda: reports.append(experiments.run_experiment(tmp_path, spec)))
    thread.start()
    deadline = time.monotonic() + 10
    while not experiments._LIVE_PROCESSES and time.monotonic() < deadline:
        time.sleep(0.05)

    started = time.monotonic()
    experiments.cancel_all()
    thread.join(10)

    assert not thread.is_alive() and time.monotonic() - started < 10
    assert reports[0]["status"] == "cancelled"
    assert reports[0]["run_count"] == 4
    assert not experiments._LIVE_PROCESSES and not experiments._LIVE_CANCELS


def test_branch_copy_keeps_symlinks_and_is_removed_when_preparation_fails(tmp_path):
    project, outside, scratch = tmp_path / "project", tmp_path / "data", tmp_path / "scratch"
    project.mkdir(); outside.mkdir(); scratch.mkdir()
    (outside / "train.bin").write_text("shared")
    (project / "config.json").write_text(json.dumps({"hidden_size": 16}))
    (project / "data").symlink_to(outside)
    (project / "latest.json").symlink_to("config.json")

    copy = scratch / "copy"
    experiments._safe_copy(project, copy, [])
    assert (copy / "data").is_symlink() and (copy / "data" / "train.bin").read_text() == "shared"
    assert (copy / "latest.json").resolve() == (copy / "config.json").resolve()

    bad = {"name": "c", "replacements": [{"file": "config.json", "find": "absent", "replace": "x"}]}
    with pytest.raises(experiments.ExperimentError):
        experiments._prepare_branch(project, scratch, {"hidden_size": 16}, "config.json", 0.5,
                                    {"hidden_size"}, ["true"], bad, [])
    assert [path.name for path in scratch.iterdir()] == ["copy"]
