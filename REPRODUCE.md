# Reproducing the analysis from raw data

Run every command from the repository root with `PYTHONPATH=.`. Outputs go to `out/` (and `share/` for files that are uploaded to Colab). Times are for a 4-core CPU with 7 GB RAM unless noted.

## Phase 2: structural audit

| Step | Command | Output | Notes |
| --- | --- | --- | --- |
| Inventory, duplicates, leakage, cross-dataset overlap | `python scripts/run_inventory.py` | `out/inventory.csv`, `out/audit_prelim.csv`, `out/cross_overlap.csv` | minutes |
| MinHash near duplicates | `python scripts/run_neardup.py` | `out/neardup_*.json` | ~30 min (winvoker) |
| Source artifacts (H3) | `python scripts/run_artifact.py` | `out/artifact_h3.json` | |
| cleanlab label-issue candidates | `python scripts/run_cleanlab.py` | `out/cleanlab_*.json`, `out/cleanlab_candidates_*.csv` | ~1 h (winvoker) |
| Embedding near duplicates (GPU) | `python scripts/build_embed_bundle.py` → `python scripts/make_notebook.py` → run `notebooks/Faz2_2_Embedding_NearDup.ipynb` on Colab → put `embed_pairs.csv.gz` into `out/` → `python scripts/run_embed_neardup.py` | `out/embed_pairs.csv.gz`, `out/embed_neardup.json` | Shortcut: `tools/prepare_from_release.py` restores the shared pairs |

## Phase 3: human validation

`scripts/build_annotation.py` draws the stratified sample (seed 20261007) and `scripts/make_annotation_xlsx.py` writes the blind annotation sheets. `run_agreement.py`, `run_final_gold.py`, `run_h3_gold.py` and `run_h3_gold_final.py` compute agreement (Cohen's κ), adjudicated gold labels and weighted error prevalence. The filled spreadsheets are not distributed (they contain raw texts). Use `tools/prepare_from_release.py`, which rebuilds `out/annotation_final.csv` from `release_data/annotation/annotations.csv` and your raw data, then run `run_h3_gold_final.py` and the later phases.

## Phase 4: models and statistics

| Step | Command | Output |
| --- | --- | --- |
| Dirty / clean / gold test definitions | `python scripts/build_splits.py` | `out/splits/<d>_s<seed>.parquet`, `out/splits/gold_<d>.parquet` (~40 min) |
| TF-IDF + LR | `python scripts/run_tfidf.py`, `python scripts/run_tfidf_wv100k.py` | `out/splits/pred*_tfidf*`, `out/tfidf_results.json` |
| BERTurk, XLM-R (GPU) | `python scripts/make_faz4_notebook.py`, zip `out/splits/*.parquet` and run `notebooks/Faz4_Transformers.ipynb` on a Colab T4 (~3 h); unzip results into `faz4_in/` | `faz4_in/pred*_{berturk,xlmr}_*.parquet` |
| Qwen3-8B zero-shot (GPU) | `notebooks/Faz4_LLM.ipynb` (reads the same Drive folder; ~1 h); unzip into `faz4_llm/` | `faz4_llm/*.parquet` |
| Evaluation | `python scripts/run_eval_models.py` | `out/model_results.csv`, `out/model_rankings.json` |
| Statistics | `python scripts/run_stats.py`, `python scripts/run_cross_len.py` | `out/stats_4_5.json`, `out/cross_len.json` |

## Phase 5: IMDB external validation

| Step | Command | Output |
| --- | --- | --- |
| Audit + classical models + Northcutt-corrected test | `python scripts/run_imdb.py` | `out/imdb_results.json`, `out/imdb_preds.parquet`, `share/imdb_colab_keys.parquet` (~5 min) |
| BERT-base, RoBERTa-base, Qwen3-8B (GPU) | `python scripts/make_faz5_notebook.py`, run `notebooks/Faz5_IMDB.ipynb` on Colab (~1.5 h; upload `share/imdb_colab_keys.parquet` when asked); unzip into `faz5_in/` | `faz5_in/*.parquet` |
| Joint analysis | `python scripts/run_imdb_tf.py` | `out/imdb_tf_results.json` |

## Determinism

All CPU steps use fixed seeds (splits 0/1/2, bootstrap seed 0, annotation sampling 20261007) and give identical results on rerun. GPU fine-tuning was not forced to be deterministic; across seeds, macro-F1 standard deviation was ≤ 0.025 (TMPR XLM-R gold) and ≤ 0.001 for IMDB accuracy. Software versions are pinned in `requirements.txt`.
