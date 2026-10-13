#!/usr/bin/env bash
# Tek komut: paylaşılan dosyalar + yerel ham veriyle makale tablolarını yeniden üretir ve doğrular.
#   bash tools/reproduce_tables.sh
set -euo pipefail
export PYTHONPATH=.
python tools/prepare_from_release.py
python tools/verify_release.py
python scripts/run_stats.py   > out/stats.log
python scripts/run_imdb_tf.py > out/imdb_tf.log
echo "Tamam: out/model_results.csv, out/stats_4_5.json, out/imdb_tf_results.json"
