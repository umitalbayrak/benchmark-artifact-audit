# Benchmark Artifact Audit

Code, audit outputs and model predictions for the paper

> **Are We Benchmarking Models or Dataset Artifacts? A Reproducible Audit Framework for Sentiment Analysis Benchmarks — Evidence from Turkish Low-Resource Benchmarks with English External Validation**
> Ümit Albayrak ({{YEAR}}). {{JOURNAL / preprint link}}
> Department of Computer Technologies, Ilgin Vocational School, Selcuk University, Konya 42600, Turkey

The framework audits a sentiment benchmark for five kinds of defects and measures how each one changes reported scores and model rankings:

| Defect | Tool in this repository |
| --- | --- |
| Exact and normalized duplicates, train–test leakage | `auditkit/core.py` (NFKC + Turkish-aware casing, punctuation removal; ASCII folding for cross-dataset matching) |
| Near duplicates | `auditkit/neardup.py` (MinHash LSH, character 5-grams, J ≥ 0.8) and multilingual sentence embeddings (cos ≥ 0.98, Colab notebook) |
| Cross-dataset contamination | `core.cross_overlap` (≥ 4-word texts) |
| Label errors | cleanlab on 5-fold out-of-fold probabilities, then stratified human validation (strata A = candidates, R = random non-candidates) with weighted prevalence |
| Source artifacts | per-source label composition and shortcut tests (`scripts/run_artifact.py`) |

Each model is evaluated on three test sets: **dirty** (the benchmark test split as is), **clean** (duplicates, leakage, near duplicates and label conflicts removed) and a **human-validated gold evaluation set** (weighted macro-F1).

## Datasets

Turkish: winvoker (TSAD), TRSAv1, TMRD, TMPR. English external validation: IMDB with the Northcutt et al. (2021) MTurk label corrections. **No benchmark text is included in this repository.** See `DATA_SHARING.md` for what is shared and why, and `data/README.md` for where to download each dataset.

## Repository layout

```
auditkit/        reusable audit library (loaders, normalization, duplicate/leakage, MinHash, TF-IDF model)
scripts/         one script per analysis step (see REPRODUCE.md for the order)
notebooks/       Colab notebooks for GPU steps (embeddings, BERTurk/XLM-R, Qwen3-8B, BERT/RoBERTa on IMDB)
tools/           prepare_from_release.py (rebuild working files from shared hashes + your raw data), verify_release.py
release_data/    hashed indices, human annotations (pseudonymous), model predictions, aggregate results
data/            put the raw datasets here (not distributed)
```

## Quick start: reproduce the paper tables without retraining

```bash
pip install -r requirements.txt
# 1. download the raw datasets into data/ as described in data/README.md
# 2. one command: rebuild working files (texts verified against SHA-1 digests), recompute all tables
#    from the shared predictions and check them against the shared results
bash tools/reproduce_tables.sh
```

To rerun everything from raw data (including training), follow `REPRODUCE.md`.

## Main results (macro-F1; mean of 3 seeds)

| Dataset | Model | Dirty | Clean | Gold |
| --- | --- | --- | --- | --- |
| winvoker | BERTurk | 0.944 | 0.944 | 0.647 |
| TMPR | BERTurk / XLM-R / TF-IDF | 0.851 / 0.814 / 0.824 | same | 0.829 / 0.839 / 0.744 |
| TRSAv1 | BERTurk | 0.835 | 0.837 | 0.842 |
| TMRD | BERTurk | 0.939 | 0.939 | 0.975 |

Leakage cleaning never changed a model ranking (Kendall τ = 1 on all four datasets), while one-sided label corruption in TMPR reversed it (dirty vs gold τ = −0.33). Full tables are in `release_data/results/`.

## License

Code: MIT (see `LICENSE`). Project-generated data in `release_data/` (gold annotations, audit measurements, model predictions, aggregate results): CC BY 4.0. Third-party benchmark content (raw texts, original labels, source fields, MTurk votes) is outside this licence and is not redistributed; original labels are rebuilt from your own copy of each dataset. See `DATA_SHARING.md`.

## Citation

See `CITATION.cff`.
