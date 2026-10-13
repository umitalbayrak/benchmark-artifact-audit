# Raw data (not included)

None of the benchmark texts are redistributed in this repository (see `../DATA_SHARING.md`). Download each dataset from its original source and place it at the paths below. The loaders in `auditkit/loaders.py` read exactly these paths.

| Dataset | Source | Expected path |
| --- | --- | --- |
| winvoker / Turkish Sentiment Dataset (CC-BY-SA-4.0) | Hugging Face: `winvoker/turkish-sentiment-analysis-dataset` (`train.csv`, `test.csv`) | `data/c704dcc2-winvoker/winvoker/{train,test}.csv` |
| TRSAv1 (Aydoğan & Kocaman, 2022) | GitHub: TRSAv1-Dataset (`TRSAv1.csv`) | `TRSAv1-Dataset/TRSAv1.csv` (repository root) |
| TMRD, Turkish movie reviews (Demirtas & Pechenizkiy, 2013) | TU Eindhoven dataset page | `data/1c7419db-Turkish_Movie_Sentiment/tr_polarity.{pos,neg}` |
| TMPR, Turkish product reviews (Demirtas & Pechenizkiy, 2013) | TU Eindhoven dataset page | `data/eeda0c8b-Turkish_Products_Sentiment/{books,dvd,electronics,kitchen}/{positive,negative}` |
| IMDB Large Movie Review Dataset v1.0 (Maas et al., 2011) | ai.stanford.edu/~amaas/data/sentiment | `data/imdb/aclImdb/` |
| IMDB label errors, MTurk validation (Northcutt et al., 2021; Apache-2.0) | GitHub: `cleanlab/label-errors` | `label-errors/mturk/imdb_mturk.json` (repository root) |

The folder names with hash prefixes (e.g. `c704dcc2-winvoker`) are kept for compatibility with the original analysis.

**Version check.** `tools/prepare_from_release.py` verifies every text against a SHA-1 digest. If a dataset version differs from the one we used (for example, rows re-ordered or edited upstream), the script stops and names the first mismatching item.
