# StraTyper Q1 pilot

This CS-GY 6513 project adds character-pattern sampling for text values in StraTyper's column type discovery (CTD) stage. The baseline uses StraTyper's length-stratified sampling. See the [original README](stratyper-main/README.md) and the [VIDA-NYU upstream repository](https://github.com/VIDA-NYU/stratyper) for the paper and original implementation. The pilot includes the Borough and Citywide tables from the NYC-E 2010–2011 Class Size dataset distributed with StraTyper, plus matching metadata (812 rows across two tables, 17 columns total).

## Setup and run

From the repository root in Windows PowerShell, with Python 3.12 and Ollama installed:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r .\stratyper-main\requirements-pilot.txt
ollama pull qwen2.5:3b
ollama list
```

Start the Ollama app before running these commands, or run `ollama serve` in a separate terminal if the server is stopped. Both configurations use the local server at `127.0.0.1:11434` for CTD, CCTA, and the judge. The pilot needs no API keys. A first run downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face and builds its own column embedding cache. On macOS or Linux, use `python3.12 -m venv .venv` and `.venv/bin/python` with the same script arguments.

```powershell
# Length baseline
& .\.venv\Scripts\python.exe .\run_nyce_pilot.py --config config/pilot_length.yaml

# Q1 character-pattern pilot (annotation only)
& .\.venv\Scripts\python.exe .\run_nyce_pilot.py --config config/pilot_character_pattern.yaml --skip-judge
```

In the two-table pilot, each method made 5 CTD calls and annotated 10 of 17 columns. The length baseline also made 10 judge calls. To run the judge for Q1, omit `--skip-judge` from the second command.

The driver resolves the config path and YAML paths from `stratyper-main/`. Both configs read `pilot/nyce_smoke/data/` and `pilot/nyce_smoke/metadata/` at the repository root, then write outputs and embedding caches to separate directories under `runs/nyce_smoke/`.

## Change Q1 settings and inspect outputs

Set `text_ctd_sample_budget` (default `10` original values per text column) and `ctd_sampling_seed` (default `42`) in `stratyper-main/config/pilot_character_pattern.yaml`. `text_ctd_sampling_type: character_pattern` enables Q1. `sampling_type: length` still applies where the original sampling path is used. The baseline uses `number_of_samples_ctd: 10` and has no Q1 sampler seed.

For each new experiment, copy the YAML to a new config such as `stratyper-main/config/pilot_character_pattern.local.yaml`, set a fresh `results_path`, and pass the new config to the driver. StraTyper reuses saved annotations and evaluations when a results directory already contains them. For a fresh embedding cache, set `embedding_path` to the same directory as `results_path`. An existing embedding cache can be reused for the same inputs and embedding model. Avoid sharing a cache across different datasets.

| Output in the run directory | What to look for |
| --- | --- |
| `logs_ctd.jsonl` | Q1 entries record the sampled original text values at `ctd_sampling.columns[*].values`, with the table, column, count, and any known-type representatives. The `prompt` field shows the values sent to CTD. |
| `all_annotations.pickle` | Full per-column predictions, keyed by table and column. Keep this file or a CSV/JSON export that also marks columns with no prediction. |
| `token_usage.txt` and phase `*.log` files | CTD and CCTA input/output token counts and elapsed seconds per phase. The token summary excludes judge tokens. Record whether embeddings were built or reused when comparing runtimes. |
| `evaluation_results.csv` and `logs_judge.jsonl` | Judge summary and individual judge calls when evaluation runs. |

## Sampling comparisons (Q2/Q3)

The pilot driver requires `sampling_type: length`. Q1 selects its text sampler separately with `text_ctd_sampling_type: character_pattern`. To include uniform random sampling, adapt the experiment entry point so the sampling change applies only to text CTD.

In the current pilot, the length baseline pools values across columns in a cluster, while Q1 samples up to 10 values from each text column. Before comparing methods at budgets of 5, 10, and 20, use the same budget unit for all three methods, count known-type representatives within that budget, and record how many values actually reach each prompt. Repeat the comparison across sampling seeds. Q1's seed of 42 controls text value selection only.

## What to share for Q5

For each run, share the exact YAML, sampling records, per-column predictions, token counts, and phase runtimes listed above. Identify the input tables, row scope, and full column list, including columns with no prediction, so coverage can be calculated. Record whether embeddings were built or reused and report any judge costs separately. For Q6, share the manually reviewed columns and any correct labels missed by the models.

Q5 pools candidate labels by table and column and uses the same evidence to judge them across methods. Keep each method's original labels so the shared judgments can be mapped back to its predictions.
