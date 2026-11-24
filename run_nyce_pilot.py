"""Run the two-table StraTyper pilot from a regular Windows PowerShell."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from time import monotonic
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "stratyper-main"
ORIGINAL_RESULTS = ROOT / "runs" / "nyce_smoke" / "original_length"
BASE_URL = "http://127.0.0.1:11434"
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
PYTHON = sys.executable
ENV = dict(os.environ, OLLAMA_BASE_URL=f"{BASE_URL}/v1", PYTHONIOENCODING="utf-8")


def run(command, phase, results):
    log_path = results / f"{phase}_{STAMP}.log"
    started = monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        header = subprocess.list2cmdline([str(part) for part in command])
        print(header, flush=True)
        print(header, file=log)
        with subprocess.Popen(
            command, cwd=UPSTREAM, env=ENV, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        duration = monotonic() - started
        message = f"Exit {status} after {duration:.2f} s. Log: {log_path}"
        print(message, file=log)
        print(message, flush=True)
    if status:
        raise subprocess.CalledProcessError(status, command)


def check_local_models(config, stages):
    with urlopen(f"{BASE_URL}/api/tags", timeout=5) as response:
        installed = {model["name"]: model for model in json.load(response)["models"]}
    for name in {config["models"][stage] for stage in stages}:
        model = installed.get(name)
        if not model or model.get("remote_host") or model.get("remote_model") or re.search(r"[:-]cloud\b", name):
            raise ValueError(f"Pilot requires an installed local Ollama model: {name}")


def verify_and_report(results, columns, had_stage_two, config):
    sys.path.insert(0, str(UPSTREAM / "src"))
    from utils import extract_dict_from_response

    with (results / "all_annotations.pickle").open("rb") as handle:
        annotations = pickle.load(handle)
    with (results / "semantic_types.pickle").open("rb") as handle:
        semantic_types = pickle.load(handle)
    predictions = []
    for table, named in annotations.items():
        for column, values in named.items():
            if (table, column) not in columns:
                raise RuntimeError(f"Unexpected annotation: {table} / {column}")
            values = [values] if isinstance(values, str) else (values or [])
            actual = [value for value in values if isinstance(value, str) and value.strip()
                      and value.strip().lower() != "none"]
            if actual:
                predictions.append((table, column, actual))
    report = [
        f"Python {sys.version.split()[0]}, Ollama model {config['models']['ctd']}",
        f"Results: {results}",
        f"Annotated columns: {len(predictions)} / {len(columns)}",
        f"Discovered types: {len(semantic_types['numeric'] | semantic_types['non_numeric'])}",
        (results / "token_usage.txt").read_text(encoding="utf-8").strip(),
    ]
    report.extend(f"{table} / {column} -> {', '.join(values)}"
                  for table, column, values in predictions[:3])
    for phase_log in sorted(results.glob("logs_*.jsonl")):
        entries = [json.loads(line) for line in phase_log.read_text(encoding="utf-8").splitlines() if line]
        invalid = sum(not extract_dict_from_response(item.get("response")) for item in entries)
        report.append(f"{phase_log.name}: {len(entries)} replies, {invalid} empty or unparseable")
    if not config.get("ccta_refinement"):
        report.append("Stage 1 CCTA refinement is disabled by the original pilot configuration.")
    if not (results / "logs_stage_2_ccta.jsonl").exists():
        if had_stage_two:
            report.append("Stage 2 CCTA reused cached annotations.")
        else:
            report.append("Stage 2 CCTA completed without an eligible candidate request.")
    print("\n".join(report), flush=True)
    (results / f"verified_baseline_{STAMP}.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not predictions:
        raise RuntimeError("all_annotations.pickle contains no substantive predictions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/pilot_length.yaml"),
                        help="Absolute path or path relative to stratyper-main.")
    parser.add_argument("--reuse-environment", action="store_true", help="Accepted for compatibility. The pilot always uses the active environment.")
    parser.add_argument("--skip-judge", action="store_true", help="Run annotation without judge evaluation.")
    args = parser.parse_args()
    config_path = (UPSTREAM / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results = (UPSTREAM / config["results_path"]).resolve()
    use_pattern = config.get("text_ctd_sampling_type") == "character_pattern"
    if config["sampling_type"] != "length" or set(config["providers"].values()) != {"ollama"}:
        raise RuntimeError("The pilot must use length sampling and local Ollama providers.")
    if use_pattern and results == ORIGINAL_RESULTS:
        raise RuntimeError("Use a separate results_path for the character-pattern pilot.")
    results.mkdir(parents=True, exist_ok=True)
    cli = [PYTHON, "-u", "src/stratyper.py", "--config", str(config_path), "--print"]
    tables = sorted((UPSTREAM / config["dataset_path"]).glob("*.csv"))
    columns = set()
    for table in tables:
        with table.open(encoding="utf-8-sig", newline="") as handle:
            columns.update((table.name, name) for name in next(csv.reader(handle)))
    stages = ("ctd", "ccta") if args.skip_judge else ("ctd", "ccta", "judge")
    check_local_models(config, stages)
    had_stage_two = (results / "annotations_stage_2.pickle").exists()
    run(cli, "character_pattern" if use_pattern else "baseline", results)
    verify_and_report(results, columns, had_stage_two, config)
    if args.skip_judge:
        return

    run([*cli, "--evaluate"], "judge", results)
    with (results / "evaluation_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not (results / "logs_judge.jsonl").is_file():
        raise RuntimeError("Judge output did not contain both results and evidence of model calls.")
    print(f"Judge produced {len(rows)} result row(s). Two-table workflow check only.", flush=True)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
