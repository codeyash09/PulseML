"""Isolated proxy-model experiments with a mandatory failure-reproduction gate.

Experiment runners are ordinary user commands, so Pulse does not assume a
framework or import the training project into its own process. Each run gets a
fresh project copy and can emit metrics by printing ``PULSE_METRICS: {json}``.
"""
from __future__ import annotations

import concurrent.futures
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_METRICS_PREFIX = "PULSE_METRICS:"
_DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".pulse", ".pulse_history", ".pulse_stream",
}
_DEFAULT_DIMENSIONS = {
    "hidden_size", "d_model", "n_embd", "intermediate_size", "d_ff",
    "num_layers", "n_layer", "num_hidden_layers", "num_attention_heads", "n_head",
}
_RELIABILITY = {
    "metric_nonfinite": 0.98,
    "metric_threshold": 0.92,
    "stderr_regex": 0.82,
    "exit_code_nonzero": 0.65,
}
_MAX_REPORT_RUNS = 120
_OUTPUT_TAIL_BYTES = 64 * 1024


class ExperimentError(ValueError):
    """An invalid or unsafe experiment specification."""


def _relative_file(root: Path, raw: str, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ExperimentError(f"{label} must be a non-empty project-relative path")
    path = Path(raw)
    if path.is_absolute() or any(part in ("..", "") for part in path.parts):
        raise ExperimentError(f"{label} must stay inside the project: {raw!r}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ExperimentError(f"{label} resolves outside the project: {raw!r}") from exc
    return resolved


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"could not read {label} {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"{label} must contain a JSON object")
    return value


def _metric(metrics: Mapping[str, Any], name: str) -> Any:
    value: Any = metrics
    for part in name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _is_failure(signature: Mapping[str, Any], metrics: Mapping[str, Any],
                stderr: str, returncode: Optional[int]) -> bool:
    kind = signature["kind"]
    if kind == "metric_nonfinite":
        value = _metric(metrics, signature["metric"])
        if value is None:
            return False
        if isinstance(value, bool):
            return False
        try:
            return not math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    if kind == "metric_threshold":
        value = _metric(metrics, signature["metric"])
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(number):
            return True
        operator = signature["operator"]
        threshold = float(signature["threshold"])
        return {
            "gt": number > threshold, "gte": number >= threshold,
            "lt": number < threshold, "lte": number <= threshold,
        }[operator]
    if kind == "stderr_regex":
        return re.search(signature["pattern"], stderr, re.IGNORECASE) is not None
    return returncode not in (None, 0)


def _validate_signature(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ExperimentError("failure_signature must be a JSON object")
    kind = raw.get("kind")
    if kind not in _RELIABILITY:
        raise ExperimentError("failure_signature.kind must be metric_nonfinite, metric_threshold, stderr_regex, or exit_code_nonzero")
    result = dict(raw)
    if kind in ("metric_nonfinite", "metric_threshold"):
        metric = raw.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            raise ExperimentError("metric failure signatures need a metric name")
        result["metric"] = metric.strip()
    if kind == "metric_threshold":
        if raw.get("operator") not in ("gt", "gte", "lt", "lte"):
            raise ExperimentError("metric_threshold operator must be gt, gte, lt, or lte")
        try:
            threshold = float(raw["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentError("metric_threshold needs a numeric threshold") from exc
        if not math.isfinite(threshold):
            raise ExperimentError("failure threshold must be finite")
        result["threshold"] = threshold
    if kind == "stderr_regex":
        pattern = raw.get("pattern")
        if not isinstance(pattern, str) or len(pattern) > 500:
            raise ExperimentError("stderr_regex needs a pattern of at most 500 characters")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ExperimentError(f"invalid failure regex: {exc}") from exc
    return result


def _set_scaled_dimensions(config: Any, factor: float,
                           selected: set[str]) -> int:
    """Scale recognized integer architecture dimensions, preserving valid head counts."""
    changed = 0
    if isinstance(config, dict):
        for key, value in list(config.items()):
            if key in selected and isinstance(value, int) and not isinstance(value, bool) and value > 1:
                scaled = max(1, int(math.floor(value * factor)))
                config[key] = scaled
                changed += scaled != value
            elif isinstance(value, (dict, list)):
                changed += _set_scaled_dimensions(value, factor, selected)
        hidden = next((config.get(key) for key in ("hidden_size", "d_model", "n_embd")
                       if isinstance(config.get(key), int) and config[key] > 0), None)
        heads_key = next((key for key in ("num_attention_heads", "n_head")
                          if isinstance(config.get(key), int) and config[key] > 0), None)
        if hidden and heads_key and hidden % config[heads_key]:
            before = config[heads_key]
            config[heads_key] = max(
                divisor for divisor in range(1, min(hidden, before) + 1)
                if hidden % divisor == 0
            )
            changed += config[heads_key] != before
    elif isinstance(config, list):
        for item in config:
            changed += _set_scaled_dimensions(item, factor, selected)
    return changed


def _safe_copy(root: Path, destination: Path, excludes: Sequence[str]) -> None:
    excluded_names = _DEFAULT_EXCLUDES | set(excludes)
    root_resolved = root.resolve()

    def ignore(directory: str, names: List[str]) -> set[str]:
        ignored = set()
        for name in names:
            source = Path(directory) / name
            if name in excluded_names or source.is_symlink():
                ignored.add(name)
            elif any(fnmatch.fnmatch(str(source.resolve().relative_to(root_resolved)), pattern) for pattern in excludes):
                ignored.add(name)
        return ignored

    shutil.copytree(root, destination, ignore=ignore, symlinks=False)


def _apply_interventions(root: Path, candidate: Mapping[str, Any]) -> List[str]:
    changed: List[str] = []
    replacements = candidate.get("replacements", [])
    if not isinstance(replacements, list):
        raise ExperimentError(f"candidate {candidate.get('name', '?')!r} replacements must be a list")
    for item in replacements:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise ExperimentError("each replacement needs file, find, and replace strings")
        path = _relative_file(root, item["file"], "replacement file")
        find, replace = item.get("find"), item.get("replace")
        if not isinstance(find, str) or not find or not isinstance(replace, str):
            raise ExperimentError("each replacement needs a non-empty find string and a replace string")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ExperimentError(f"could not read replacement file {item['file']!r}: {exc}") from exc
        expected = item.get("expected_count", 1)
        if not isinstance(expected, int) or expected < 1:
            raise ExperimentError("replacement expected_count must be a positive integer")
        actual = text.count(find)
        if actual != expected:
            raise ExperimentError(
                f"candidate {candidate.get('name', '?')!r}: {item['file']!r} expected "
                f"{expected} occurrence(s) of its target, found {actual}"
            )
        path.write_text(text.replace(find, replace), encoding="utf-8")
        changed.append(item["file"])
    return changed


def _tail(file_obj: Any) -> str:
    file_obj.seek(0, os.SEEK_END)
    end = file_obj.tell()
    file_obj.seek(max(0, end - _OUTPUT_TAIL_BYTES))
    return file_obj.read().decode("utf-8", errors="replace")


def _scan_metrics(file_obj: Any, signature: Mapping[str, Any]) -> Dict[str, Any]:
    """Read every metric record without retaining an unbounded training history."""
    file_obj.seek(0)
    latest: Dict[str, Any] = {}
    summary: Dict[str, Dict[str, Any]] = {}
    samples = 0
    failure_observed = False
    evidence_complete = signature["kind"] not in ("metric_nonfinite", "metric_threshold")
    for raw_line in file_obj:
        if not raw_line.startswith(_METRICS_PREFIX.encode("ascii")):
            continue
        try:
            snapshot = json.loads(raw_line[len(_METRICS_PREFIX):].decode("utf-8", errors="replace").strip())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(snapshot, dict):
            continue
        samples += 1
        latest = snapshot
        if signature["kind"] in ("metric_nonfinite", "metric_threshold"):
            evidence_complete |= _metric(snapshot, signature["metric"]) is not None
            failure_observed |= _is_failure(signature, snapshot, "", None)
        for name, value in snapshot.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (OverflowError, ValueError):
                continue
            item = summary.setdefault(name, {
                "samples": 0, "finite_samples": 0, "nonfinite_observed": False,
                "first": None, "last": None, "minimum": None, "maximum": None,
            })
            item["samples"] += 1
            if not math.isfinite(number):
                item["nonfinite_observed"] = True
                continue
            item["finite_samples"] += 1
            if item["first"] is None:
                item["first"] = number
            item["last"] = number
            item["minimum"] = number if item["minimum"] is None else min(item["minimum"], number)
            item["maximum"] = number if item["maximum"] is None else max(item["maximum"], number)
    return {
        "metrics": latest, "metrics_samples": samples, "metric_summary": summary,
        "failure_observed": failure_observed, "evidence_complete": evidence_complete,
    }


def _stderr_matches(file_obj: Any, signature: Mapping[str, Any]) -> bool:
    if signature["kind"] != "stderr_regex":
        return False
    pattern = re.compile(signature["pattern"], re.IGNORECASE)
    file_obj.seek(0)
    return any(pattern.search(line.decode("utf-8", errors="replace")) for line in file_obj)


def _combine_metric_summaries(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    combined: Dict[str, Any] = {}
    names = {name for summary in summaries for name in summary}
    for name in names:
        entries = [summary[name] for summary in summaries if name in summary]
        minima = [entry["minimum"] for entry in entries if entry.get("minimum") is not None]
        maxima = [entry["maximum"] for entry in entries if entry.get("maximum") is not None]
        combined[name] = {
            "samples": sum(entry.get("samples", 0) for entry in entries),
            "finite_samples": sum(entry.get("finite_samples", 0) for entry in entries),
            "nonfinite_observed": any(entry.get("nonfinite_observed", False) for entry in entries),
            "first": next((entry.get("first") for entry in entries if entry.get("first") is not None), None),
            "last": next((entry.get("last") for entry in reversed(entries) if entry.get("last") is not None), None),
            "minimum": min(minima) if minima else None,
            "maximum": max(maxima) if maxima else None,
        }
    return combined


def _dimension_snapshot(config: Any, dimensions: set[str], prefix: str = "") -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    if isinstance(config, dict):
        for key, value in config.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in dimensions and isinstance(value, (int, float)) and not isinstance(value, bool):
                found[path] = value
            elif isinstance(value, (dict, list)):
                found.update(_dimension_snapshot(value, dimensions, path))
    elif isinstance(config, list):
        for index, value in enumerate(config):
            if isinstance(value, (dict, list)):
                found.update(_dimension_snapshot(value, dimensions, f"{prefix}[{index}]"))
    return found


def _run_command(command: Sequence[str], cwd: Path, env: Mapping[str, str],
                 timeout: float, cancel: threading.Event,
                 signature: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.monotonic()
    result: Dict[str, Any] = {
        "duration_seconds": 0.0, "returncode": None, "timed_out": False,
        "cancelled": False, "launch_error": None, "metrics": {}, "metrics_samples": 0,
        "metric_summary": {}, "failure_observed": False, "evidence_complete": False,
        "failure_detail": None,
    }
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            process = subprocess.Popen(
                list(command), cwd=str(cwd), env=dict(env), stdout=out, stderr=err,
                stdin=subprocess.DEVNULL, shell=False,
                start_new_session=os.name != "nt",
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except OSError as exc:
            result["launch_error"] = f"{type(exc).__name__}: {exc}"
            return result
        while process.poll() is None:
            if cancel.is_set():
                result["cancelled"] = True
                _terminate_process(process)
                break
            elif time.monotonic() - started >= timeout:
                result["timed_out"] = True
                _terminate_process(process)
                break
            time.sleep(0.05)
        result["returncode"] = process.returncode
        stderr = _tail(err)
        result.update(_scan_metrics(out, signature))
        result["failure_observed"] |= _stderr_matches(err, signature)
        if signature["kind"] == "exit_code_nonzero":
            result["failure_observed"] = process.returncode not in (None, 0)
            result["evidence_complete"] = True
        result["duration_seconds"] = round(time.monotonic() - started, 4)
        if result["launch_error"] is None and process.returncode and not stderr:
            result["failure_detail"] = f"process exited with code {process.returncode}"
        else:
            result["failure_detail"] = stderr[-2000:] or None
        result["_stderr"] = stderr
    return result


def _terminate_process(process: subprocess.Popen) -> None:
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            if process.poll() is None:
                process.terminate()
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
        else:
            process.kill()
        process.wait()


def _sanitize_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    try:
        from pulse.pulse_supabase import scrub_secrets
        return scrub_secrets(text)
    except ImportError:
        return text


def _validate_spec(spec: Mapping[str, Any], root: Path) -> Dict[str, Any]:
    command = spec.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(part, str) and part for part in command):
        raise ExperimentError("command must be a non-empty argument array; shell strings are not accepted")
    evidence = spec.get("original_evidence")
    if not isinstance(evidence, str) or len(evidence.strip()) < 8:
        raise ExperimentError("original_evidence must describe the observed live-run failure in at least 8 characters")
    signature = _validate_signature(spec.get("failure_signature"))
    original_metrics = spec.get("original_metrics", [])
    if signature["kind"] == "metric_nonfinite":
        if not isinstance(original_metrics, list) or not any(
            isinstance(value, float) and not math.isfinite(value) for value in original_metrics
        ):
            raise ExperimentError("metric_nonfinite requires a non-finite original_metrics value from the live run")
    if signature["kind"] == "metric_threshold":
        if not isinstance(original_metrics, list) or not original_metrics:
            raise ExperimentError("metric_threshold requires original_metrics from the failed live run")
        observed = False
        for value in original_metrics:
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            operator, threshold = signature["operator"], signature["threshold"]
            observed |= ({"gt": number > threshold, "gte": number >= threshold,
                          "lt": number < threshold, "lte": number <= threshold}[operator])
        if not observed:
            raise ExperimentError("original_metrics do not meet the configured failure threshold")
    config_file = spec.get("proxy", {}).get("config_file") if isinstance(spec.get("proxy"), dict) else None
    if not isinstance(config_file, str):
        raise ExperimentError("proxy.config_file must name the model's JSON architecture/training config")
    config_path = _relative_file(root, config_file, "proxy.config_file")
    if not config_path.is_file():
        raise ExperimentError(f"proxy config does not exist: {config_file}")
    config = _read_json_object(config_path, "proxy config")
    scales = spec.get("proxy", {}).get("scales", [0.25, 0.5, 0.75])
    if not isinstance(scales, list) or not scales:
        raise ExperimentError("proxy.scales must be a non-empty list of scale factors")
    normalized_scales = []
    for scale in scales:
        try:
            scale = float(scale)
        except (TypeError, ValueError) as exc:
            raise ExperimentError("proxy scales must be numbers between 0 and 1") from exc
        if not math.isfinite(scale) or scale <= 0 or scale >= 1:
            raise ExperimentError("proxy scales must be greater than 0 and less than 1")
        if scale not in normalized_scales:
            normalized_scales.append(scale)
    normalized_scales.sort()
    dimensions = spec.get("proxy", {}).get("dimension_keys", sorted(_DEFAULT_DIMENSIONS))
    if not isinstance(dimensions, list) or not all(isinstance(key, str) for key in dimensions):
        raise ExperimentError("proxy.dimension_keys must be an array of JSON key names")
    if not any(isinstance(node, dict) and any(
        key in node and isinstance(node[key], int) and node[key] > 1 for key in dimensions
    ) for node in _walk_dicts(config)):
        raise ExperimentError("proxy config has no recognized scalable integer dimension")
    repetitions = spec.get("repetitions", 2)
    workers = spec.get("max_parallel", 2)
    timeout = spec.get("timeout_seconds", 600)
    for name, value, low, high in (
        ("repetitions", repetitions, 1, 20), ("max_parallel", workers, 1, 32),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ExperimentError(f"{name} must be an integer from {low} to {high}")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ExperimentError("timeout_seconds must be a positive finite number")
    adaptive_repetitions = spec.get("adaptive_extra_repetitions", 0)
    adaptive_top = spec.get("adaptive_top", 2)
    if isinstance(adaptive_repetitions, bool) or not isinstance(adaptive_repetitions, int) or not 0 <= adaptive_repetitions <= 20:
        raise ExperimentError("adaptive_extra_repetitions must be an integer from 0 to 20")
    if isinstance(adaptive_top, bool) or not isinstance(adaptive_top, int) or not 0 <= adaptive_top <= 32:
        raise ExperimentError("adaptive_top must be an integer from 0 to 32")
    candidates = spec.get("candidates", [])
    if not isinstance(candidates, list):
        raise ExperimentError("candidates must be an array")
    names = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("name"), str) or not candidate["name"].strip():
            raise ExperimentError("each candidate needs a non-empty name")
        if candidate["name"] in names or candidate["name"] == "control":
            raise ExperimentError("candidate names must be unique and 'control' is reserved")
        names.add(candidate["name"])
        if not isinstance(candidate.get("hypothesis", ""), str):
            raise ExperimentError("candidate hypothesis must be a string")
        if not candidate.get("hypothesis", "").strip():
            raise ExperimentError(f"candidate {candidate['name']!r} needs a hypothesis")
        if not isinstance(candidate.get("replacements", []), list):
            raise ExperimentError("candidate replacements must be an array")
        for replacement in candidate.get("replacements", []):
            if not isinstance(replacement, dict) or not isinstance(replacement.get("file"), str):
                raise ExperimentError(f"candidate {candidate['name']!r} has a malformed replacement")
            _relative_file(root, replacement["file"], "replacement file")
            if not isinstance(replacement.get("find"), str) or not replacement["find"]:
                raise ExperimentError(f"candidate {candidate['name']!r} replacement find must be non-empty")
            if not isinstance(replacement.get("replace"), str):
                raise ExperimentError(f"candidate {candidate['name']!r} replacement replace must be a string")
            expected_count = replacement.get("expected_count", 1)
            if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
                raise ExperimentError(f"candidate {candidate['name']!r} expected_count must be a positive integer")
    objective = spec.get("objective", {})
    if objective and (not isinstance(objective, dict) or not isinstance(objective.get("metric"), str)
                      or objective.get("direction") not in ("min", "max")):
        raise ExperimentError("objective needs metric and direction ('min' or 'max')")
    minimum_confidence = spec.get("minimum_reproduction_confidence", 0.8)
    if isinstance(minimum_confidence, bool) or not isinstance(minimum_confidence, (int, float)) or not 0 < minimum_confidence <= 1:
        raise ExperimentError("minimum_reproduction_confidence must be in (0, 1]")
    excludes = spec.get("copy_excludes", [])
    if not isinstance(excludes, list) or not all(isinstance(pattern, str) for pattern in excludes):
        raise ExperimentError("copy_excludes must be an array of path patterns")
    environment = spec.get("env", {})
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in environment.items()
    ):
        raise ExperimentError("env must map strings to strings")
    resource_class = spec.get("resource_class", "auto")
    if resource_class not in ("auto", "cpu", "gpu"):
        raise ExperimentError("resource_class must be auto, cpu, or gpu")
    devices = spec.get("devices", [])
    if not isinstance(devices, list) or not all(isinstance(device, (str, int)) and not isinstance(device, bool)
                                               for device in devices):
        raise ExperimentError("devices must be an array of string or integer identifiers")
    if resource_class == "gpu" and not devices:
        raise ExperimentError("resource_class gpu requires explicit devices for safe scheduling")
    validation_scales = spec.get("validation_scales", [1.0])
    if not isinstance(validation_scales, list) or not validation_scales:
        raise ExperimentError("validation_scales must be a non-empty array")
    for scale in validation_scales:
        try:
            numeric_scale = float(scale)
        except (TypeError, ValueError) as exc:
            raise ExperimentError("validation scales must be numbers in (0, 1]") from exc
        if not math.isfinite(numeric_scale) or numeric_scale <= 0 or numeric_scale > 1:
            raise ExperimentError("validation scales must be in (0, 1]")
    validation_scales = sorted({float(scale) for scale in validation_scales} | {1.0})
    validate_top = spec.get("validate_top", 1)
    if isinstance(validate_top, bool) or not isinstance(validate_top, int) or validate_top < 0:
        raise ExperimentError("validate_top must be a non-negative integer")
    return {
        "command": command, "signature": signature, "evidence": evidence.strip(),
        "original_metrics": original_metrics,
        "config_file": config_file, "config": config, "scales": normalized_scales,
        "dimension_keys": set(dimensions), "repetitions": repetitions,
        "workers": workers, "timeout": float(timeout), "candidates": candidates,
        "objective": objective, "minimum_confidence": float(minimum_confidence),
        "seeds": spec.get("seeds"),
        "adaptive_repetitions": adaptive_repetitions,
        "adaptive_top": adaptive_top,
        "excludes": excludes,
        "devices": devices,
        "validation_scales": validation_scales,
        "validate_top": validate_top,
    }


def _walk_dicts(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _prepare_branch(root: Path, temp_root: Path, config: Mapping[str, Any],
                    config_file: str, factor: float, dimensions: set[str],
                    command: Sequence[str], candidate: Optional[Mapping[str, Any]],
                    excludes: Sequence[str]) -> Tuple[Path, List[str], Dict[str, Any]]:
    branch = temp_root / uuid.uuid4().hex
    _safe_copy(root, branch, excludes)
    proxy_config = json.loads(json.dumps(config))
    changed = _set_scaled_dimensions(proxy_config, factor, dimensions)
    destination = _relative_file(branch, config_file, "proxy config")
    destination.write_text(json.dumps(proxy_config, indent=2) + "\n", encoding="utf-8")
    edits = _apply_interventions(branch, candidate) if candidate else []
    expanded = [part.replace("{project}", str(branch)).replace("{config}", str(destination))
                for part in command]
    return branch, edits, {
        "dimension_changes": changed, "config": proxy_config, "command": expanded,
    }


def run_experiment(project_root: str | os.PathLike[str], spec: Mapping[str, Any],
                   cancel_event: Optional[threading.Event] = None) -> Dict[str, Any]:
    """Run a proxy experiment; no candidate is tested until the baseline gate passes.

    The runner is deliberately framework-neutral. Training code emits metric
    snapshots as ``PULSE_METRICS: {"loss": 0.2, "grad_norm": 1.4}`` lines.
    """
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ExperimentError(f"project directory does not exist: {root}")
    settings = _validate_spec(spec, root)
    cancel = cancel_event or threading.Event()
    run_records: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "type": "experiment", "id": uuid.uuid4().hex[:12],
        "status": "inconclusive", "stage": "reproduction_gate",
        "failure_signature": settings["signature"],
        "original_evidence": settings["evidence"],
        "original_metrics": settings["original_metrics"],
        "proxy_attempts": [], "branches": [], "validation": [],
        "candidate_count": len(settings["candidates"]),
    }
    base_env = os.environ.copy()
    if isinstance(spec.get("env"), dict):
        for key, value in spec["env"].items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ExperimentError("env must map strings to strings")
            base_env[key] = value
    resource = spec.get("resource_class", "auto")
    devices = settings["devices"]
    if resource == "cpu":
        workers = settings["workers"]
    elif devices:
        workers = min(settings["workers"], len(devices))
    else:
        workers = 1
    workers = max(1, workers)
    seeds = settings["seeds"]
    if seeds is None:
        seeds = list(range(settings["repetitions"] + settings["adaptive_repetitions"]))
    if not isinstance(seeds, list) or not seeds or len(seeds) < settings["repetitions"] + settings["adaptive_repetitions"]:
        raise ExperimentError("seeds must provide one value per repetition, including adaptive extra repetitions")
    try:
        with tempfile.TemporaryDirectory(prefix="pulse-experiment-") as scratch:
            temp_root = Path(scratch)

            def one_run(label: str, factor: float, seed: Any,
                        candidate: Optional[Mapping[str, Any]], index: int) -> Dict[str, Any]:
                if cancel.is_set():
                    return {"branch": label, "seed": seed, "factor": factor, "cancelled": True}
                branch = None
                try:
                    branch, edited, prepared = _prepare_branch(
                        root, temp_root, settings["config"], settings["config_file"],
                        factor, settings["dimension_keys"], settings["command"], candidate,
                        settings["excludes"],
                    )
                    env = dict(base_env)
                    env["PULSE_EXPERIMENT_SEED"] = str(seed)
                    env["PULSE_EXPERIMENT_ID"] = report["id"]
                    env["PULSE_EXPERIMENT_BRANCH"] = label
                    if devices and resource != "cpu":
                        seed_key = int.from_bytes(
                            hashlib.sha256(str(seed).encode("utf-8")).digest()[:8], "big",
                        ) % len(devices)
                        device = str(devices[seed_key])
                        env["CUDA_VISIBLE_DEVICES"] = device
                    else:
                        device = None
                    command = [part.replace("{seed}", str(seed)).replace("{branch}", label)
                               for part in prepared["command"]]
                    outcome = _run_command(
                        command, branch, env, settings["timeout"], cancel,
                        settings["signature"],
                    )
                    error = outcome.pop("failure_detail", None)
                    outcome.pop("_stderr", None)
                    return {
                        "branch": label, "seed": seed, "factor": factor,
                        "device": device, "duration_seconds": outcome["duration_seconds"],
                        "returncode": outcome["returncode"], "timed_out": outcome["timed_out"],
                        "cancelled": outcome["cancelled"], "launch_error": outcome["launch_error"],
                        "metrics": outcome["metrics"],
                        "failure_observed": outcome["failure_observed"],
                        "metrics_samples": outcome["metrics_samples"],
                        "metric_summary": outcome["metric_summary"],
                        "evidence_complete": outcome["evidence_complete"],
                        "failure_detail": _sanitize_text(error) if error else None,
                        "edits": edited, "dimension_changes": prepared["dimension_changes"],
                        "proxy_dimensions": _dimension_snapshot(
                            prepared["config"], settings["dimension_keys"],
                        ),
                    }
                except Exception as exc:
                    return {
                        "branch": label, "seed": seed, "factor": factor,
                        "returncode": None, "timed_out": False, "cancelled": False,
                        "launch_error": f"{type(exc).__name__}: {exc}",
                        "failure_detail": _sanitize_text(str(exc)), "failure_observed": False,
                        "metrics": {}, "duration_seconds": 0.0, "edits": [],
                    }
                finally:
                    if branch is not None:
                        shutil.rmtree(branch, ignore_errors=True)

            reproduction: Optional[Dict[str, Any]] = None
            for factor in settings["scales"]:
                if cancel.is_set():
                    report["status"] = "cancelled"
                    break
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(one_run, "baseline", factor, seeds[i], None, i)
                        for i in range(settings["repetitions"])
                    ]
                    trials = [future.result() for future in futures]
                run_records.extend(trials)
                hits = sum(record.get("failure_observed", False) for record in trials)
                consistency = hits / max(1, len(trials))
                confidence = round(
                    consistency * _RELIABILITY[settings["signature"]["kind"]]
                    if hits == len(trials) else consistency * _RELIABILITY[settings["signature"]["kind"]] * 0.9,
                    3,
                )
                attempt = {
                    "scale": factor, "trials": len(trials), "reproduced": hits,
                    "consistency": round(consistency, 3), "confidence": confidence,
                    "dimension_changes": trials[0].get("dimension_changes", 0) if trials else 0,
                    "metrics": [record.get("metrics", {}) for record in trials],
                    "proxy_dimensions": trials[0].get("proxy_dimensions", {}) if trials else {},
                }
                report["proxy_attempts"].append(attempt)
                if cancel.is_set():
                    report["status"] = "cancelled"
                    break
                if confidence >= settings["minimum_confidence"]:
                    reproduction = attempt
                    break
            report["reproduction"] = reproduction or {
                "passed": False, "minimum_confidence": settings["minimum_confidence"],
                "message": "No proxy scale reproduced the configured failure consistently enough.",
            }
            if reproduction:
                report["reproduction"]["passed"] = True
                report["reproduction"]["mechanism"] = settings["signature"]["kind"]
                report["reproduction"]["scale"] = reproduction["scale"]
                report["reproduction"]["explanation"] = (
                    "The same configured failure signal was observed in repeated unmodified "
                    "proxy runs. This is evidence about that signal, not proof the proxy preserves "
                    "the full model's failure mechanism."
                )
                report["stage"] = "parallel_candidates"
                factor = reproduction["scale"]
                branches = [{"name": "control", "hypothesis": "Unchanged proxy control",
                             "candidate": None}] + [
                    {"name": item["name"], "hypothesis": item.get("hypothesis", ""),
                     "candidate": item} for item in settings["candidates"]
                ]
                tasks = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    for branch_index, branch in enumerate(branches):
                        for i in range(settings["repetitions"]):
                            tasks.append(pool.submit(
                                one_run, branch["name"], factor, seeds[i],
                                branch["candidate"], branch_index * settings["repetitions"] + i,
                            ))
                    results = [future.result() for future in tasks]
                run_records.extend(results)
                if cancel.is_set():
                    report["status"] = "cancelled"
                    report["stage"] = "parallel_candidates"
                    report["conclusion"] = "Experiment cancelled during candidate evaluation; partial results are not ranked."
                    report["run_count"] = len(run_records)
                    report["runs"] = run_records[-_MAX_REPORT_RUNS:]
                    return report
                objective = settings["objective"]
                adaptive_allocations = []
                if settings["adaptive_repetitions"] and settings["adaptive_top"]:
                    control_trials = [r for r in results if r["branch"] == "control"]
                    control_success = sum(
                        not r.get("failure_observed") and not r.get("launch_error")
                        and r.get("returncode") == 0 and not r.get("timed_out")
                        and not r.get("cancelled") for r in control_trials
                    ) / max(1, len(control_trials))
                    ranked_early = []
                    control_objective = []
                    if objective:
                        for record in control_trials:
                            value = _metric(record.get("metrics", {}), objective["metric"])
                            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                                control_objective.append(float(value))
                    control_mean = sum(control_objective) / len(control_objective) if control_objective else None
                    for candidate in settings["candidates"]:
                        selected = [r for r in results if r["branch"] == candidate["name"]]
                        successes = sum(
                            not r.get("failure_observed") and not r.get("launch_error")
                            and r.get("evidence_complete", False)
                            and r.get("returncode") == 0 and not r.get("timed_out")
                            and not r.get("cancelled") for r in selected
                        )
                        score = successes / max(1, len(selected))
                        values = []
                        if objective:
                            for record in selected:
                                value = _metric(record.get("metrics", {}), objective["metric"])
                                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                                    values.append(float(value))
                        mean = sum(values) / len(values) if values else None
                        gain = None
                        if mean is not None and control_mean is not None:
                            gain = control_mean - mean if objective["direction"] == "min" else mean - control_mean
                        if score > control_success or (gain is not None and gain > 0):
                            ranked_early.append((score, gain or 0.0, candidate))
                    ranked_early.sort(key=lambda item: (item[0], item[1]), reverse=True)
                    extra_targets = [{"name": "control", "candidate": None}] + [
                        {"name": candidate["name"], "candidate": candidate}
                        for _score, _gain, candidate in ranked_early[:settings["adaptive_top"]]
                    ]
                    adaptive_results = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = []
                        for target_index, target in enumerate(extra_targets):
                            for offset in range(settings["adaptive_repetitions"]):
                                seed_index = settings["repetitions"] + offset
                                futures.append(pool.submit(
                                    one_run, target["name"], factor, seeds[seed_index],
                                    target["candidate"], target_index * settings["adaptive_repetitions"] + offset,
                                ))
                        adaptive_results = [future.result() for future in futures]
                    results.extend(adaptive_results)
                    run_records.extend(adaptive_results)
                    adaptive_allocations = [
                        {"branch": target["name"], "additional_trials": settings["adaptive_repetitions"]}
                        for target in extra_targets
                    ]
                report["adaptive_allocations"] = adaptive_allocations
                baseline_values = []
                if objective:
                    for record in results:
                        if record.get("branch") == "control":
                            value = _metric(record.get("metrics", {}), objective["metric"])
                            if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
                                baseline_values.append(float(value))
                baseline_mean = sum(baseline_values) / len(baseline_values) if baseline_values else None
                for branch in branches:
                    selected = [record for record in results if record.get("branch") == branch["name"]]
                    successes = sum(not record.get("failure_observed", False) and
                                    record.get("evidence_complete", False) and
                                    not record.get("launch_error") and
                                    record.get("returncode") == 0 and
                                    not record.get("timed_out") and not record.get("cancelled")
                                    for record in selected)
                    metrics = []
                    if objective:
                        metrics = [float(value) for record in selected
                                   if isinstance((value := _metric(record.get("metrics", {}), objective["metric"])),
                                                 (int, float)) and not isinstance(value, bool)
                                   and math.isfinite(value)]
                    mean_metric = sum(metrics) / len(metrics) if metrics else None
                    improvement = None
                    if mean_metric is not None and baseline_mean is not None:
                        improvement = round(
                            baseline_mean - mean_metric if objective["direction"] == "min"
                            else mean_metric - baseline_mean, 8,
                        )
                    score = successes / max(1, len(selected))
                    if branch["name"] != "control" and improvement is not None:
                        score += max(-1.0, min(1.0, improvement / max(abs(baseline_mean), 1e-12))) * 0.25
                    report["branches"].append({
                        "name": branch["name"], "hypothesis": branch["hypothesis"],
                        "trials": len(selected), "successful_trials": successes,
                        "success_rate": round(successes / max(1, len(selected)), 3),
                        "mean_metric": round(mean_metric, 8) if mean_metric is not None else None,
                        "baseline_mean_metric": round(baseline_mean, 8) if baseline_mean is not None else None,
                        "objective_improvement": improvement, "score": round(score, 5),
                        "resource_seconds": round(sum(r.get("duration_seconds", 0) for r in selected), 3),
                        "edits": sorted({path for r in selected for path in r.get("edits", [])}),
                        "metrics": [r.get("metrics", {}) for r in selected],
                        "metric_summary": _combine_metric_summaries(
                            [r.get("metric_summary", {}) for r in selected]
                        ),
                        "failures": [r.get("failure_detail") for r in selected if r.get("failure_detail")],
                    })
                ranked = sorted((branch for branch in report["branches"] if branch["name"] != "control"),
                                key=lambda item: (item["success_rate"], item["score"]), reverse=True)
                report["ranking"] = [item["name"] for item in ranked]
                promising = [item for item in ranked if item["success_rate"] == 1.0]
                validation_scales = settings["validation_scales"]
                if not isinstance(validation_scales, list) or not validation_scales:
                    raise ExperimentError("validation_scales must be a non-empty array")
                limit = settings["validate_top"]
                if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                    raise ExperimentError("validate_top must be a non-negative integer")
                for candidate_name in [item["name"] for item in promising[:limit]]:
                    candidate = next(item for item in settings["candidates"] if item["name"] == candidate_name)
                    for validation_scale in validation_scales:
                        validation_scale = float(validation_scale)
                        if not math.isfinite(validation_scale) or validation_scale <= 0 or validation_scale > 1:
                            raise ExperimentError("validation scales must be in (0, 1]")
                        validation_trials = []
                        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                            futures = [pool.submit(one_run, candidate_name, validation_scale, seeds[i],
                                                   candidate, i) for i in range(settings["repetitions"])]
                            validation_trials = [future.result() for future in futures]
                        run_records.extend(validation_trials)
                        hits = sum(r.get("failure_observed", False) for r in validation_trials)
                        result = {
                            "candidate": candidate_name, "scale": validation_scale,
                            "trials": len(validation_trials), "failures_observed": hits,
                            "passed": hits == 0 and all(
                                r.get("evidence_complete", False) and not r.get("launch_error")
                                and r.get("returncode") == 0 and not r.get("timed_out") and not r.get("cancelled")
                                for r in validation_trials
                            ),
                            "scope": "isolated source-scale copy" if validation_scale == 1.0 else "isolated proxy copy",
                        }
                        report["validation"].append(result)
                        if cancel.is_set():
                            break
                    if cancel.is_set():
                        break
                validated = {item["candidate"] for item in report["validation"] if item["passed"]}
                for branch in report["branches"]:
                    if branch["name"] != "control":
                        branch["validated"] = branch["name"] in validated
                report["status"] = "cancelled" if cancel.is_set() else "completed"
                report["stage"] = "cancelled" if cancel.is_set() else "validation"
                report["resource_seconds"] = round(sum(r.get("duration_seconds", 0) for r in run_records), 3)
                if cancel.is_set():
                    report["conclusion"] = "Experiment cancelled; partial branch results are not a final comparison."
                elif not ranked:
                    report["conclusion"] = "No intervention candidates were supplied."
                elif validated:
                    report["conclusion"] = (
                        "Candidate(s) passed on isolated source-scale copies. This does not prove a "
                        "fix in the live run; use Pulse's approval and verification flow before applying."
                    )
                elif promising:
                    report["conclusion"] = (
                        "Candidate(s) passed the proxy, but source-scale validation did not pass. "
                        "Do not apply based on proxy evidence alone."
                    )
                else:
                    report["conclusion"] = "No candidate eliminated the reproduced failure in every trial."
            elif report["status"] != "cancelled":
                report["status"] = "inconclusive"
                report["stage"] = "reproduction_gate"
                report["conclusion"] = (
                    "Candidate experiments were not run because the unmodified proxy did not meet "
                    "the failure-reproduction confidence threshold."
                )
            report["run_count"] = len(run_records)
            if len(run_records) > _MAX_REPORT_RUNS:
                report["runs_truncated"] = True
            report["runs"] = run_records[-_MAX_REPORT_RUNS:]
            try:
                from pulse.pulse_supabase import scrub_secrets
                report = json.loads(scrub_secrets(json.dumps(report, ensure_ascii=True)))
            except ImportError:
                pass
            return report
    except BaseException:
        cancel.set()
        raise
