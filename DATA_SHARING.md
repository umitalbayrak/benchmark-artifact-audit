# Data sharing statement

This file states, item by item, what this repository shares and why. **Status: revised after the licence check (phase 6.2). Only the ethics wording in §4 remains to be confirmed.**

## 1. Raw benchmark texts: not shared

| Dataset | Licence / terms found | Redistribution in this repository |
| --- | --- | --- |
| winvoker (TSAD) | Hugging Face card: CC-BY-SA-4.0, but compiled from heterogeneous third-party sources (Kaggle product reviews, a Twitter sentiment corpus without a verifiable licence, HUMIR, Turkish Wikipedia under CC BY-SA/GFDL) | **No texts, no original labels** |
| TRSAv1 | README: "can be used … by citing the related study"; no formal licence | **No texts, no original labels** |
| TMRD, TMPR | Publicly distributed, but no standard licence could be verified | **No texts, no original labels** |
| IMDB v1.0 | README asks for citation of Maas et al. (2011); no explicit licence | **No texts, no original labels** |
| Northcutt et al. label errors (`cleanlab/label-errors`) | Apache-2.0 | Not copied; users download it. We only use review IDs and vote counts |

Original benchmark labels and source fields are also withheld; `tools/prepare_from_release.py` rebuilds them from the user's own copy. All shared files reference texts by **row index in the original file** (`orig_idx`, after the loader in `auditkit/loaders.py`) and/or a **truncated SHA-1 digest** (first 16 hex characters) of the raw text or of its normalized form. A digest cannot be used to recover a text; it only verifies that a user's own copy matches ours.

## 2. Shared derived data

| Path | Content | Personal data? | Licence |
| --- | --- | --- | --- |
| `release_data/annotation/annotations.csv` | 994 human-validated items: item ID, dataset, row index, stratum (A/R/E/H), text digests, two independent annotator labels, adjudication verdict and final gold label (no original label) | Annotators are pseudonymous (`annotator_1`, `annotator_2`); no names, IDs or timestamps. Texts are not included, so reviewer-written personal information (names, usernames) in reviews is not shared | CC BY 4.0 |
| `release_data/annotation/annotation_population.json` | Population sizes per stratum (weights) | No | CC BY 4.0 |
| `release_data/splits/*.csv.gz` | For each dataset and seed: `orig_idx`, text digest, train/test part, clean-test drop reason (no benchmark label) | No | CC BY 4.0 (split assignment and audit decisions only) |
| `release_data/splits/gold_*.csv` | Gold evaluation sets: item ID, digest, project gold label, stratum, weight | No | CC BY 4.0 |
| `release_data/embeddings/` | Embedding near-duplicate pairs as integer IDs + cosine; digest per ID | No | CC BY 4.0 |
| `release_data/predictions/` | Model predictions, keyed by row position in the split's test part, by gold item order, or by IMDB review ID / text digest. IMDB files keep only the correction status (flipped / dropped / kept), not the original or corrected label | No | CC BY 4.0 |
| `release_data/results/` | Aggregate statistics (JSON/CSV) | No | CC BY 4.0 |

Not shared: the annotation spreadsheets (`Etiketleme_A.xlsx`, `Etiketleme_B.xlsx`, `Uzlasma.xlsx`), example files with raw texts (`*_examples.csv`, `cleanlab_candidates_*.csv`, `disagreements.csv`) and the Colab input bundles, because they contain raw texts.

## 3. Licence scope

CC BY 4.0 covers only project-generated content (gold annotations, audit measurements, predictions, aggregate results). Third-party benchmark content (texts, original labels, source fields) and the Northcutt et al. MTurk votes (Apache-2.0, downloaded by the user) are explicitly outside it. Code is MIT.

## 4. Human annotation

Two annotators labelled each item blind to the original label and to each other (5 classes: positive / negative / neutral / ambiguous / off-topic); disagreements were adjudicated. The two annotators were unpaid volunteers (native Turkish-speaking academic staff members, not part of the research team and not crowdworkers) who agreed verbally to take part and later confirmed this in a written statement of voluntary participation kept by the author. They labelled publicly available benchmark texts; they were not research subjects, and no personal data about them was collected or is released. {{ETHICS_APPROVAL_OR_EXEMPTION}}

## 5. Reproducibility levels

1. **Recompute paper tables from shared predictions** (no GPU, minutes): `bash tools/reproduce_tables.sh`.
2. **Rerun the audit and classical models from raw data** (CPU, hours): `REPRODUCE.md` steps 1–4.
3. **Retrain transformers / rerun the LLM** (Colab T4 GPU): notebooks in `notebooks/`. Results will match up to GPU non-determinism (seed std ≤ 0.025 macro-F1 in our runs).
