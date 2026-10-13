"""Yeniden üretilebilirlik kontrolü: prepare_from_release.py sonrası makale tablolarını tahminlerden yeniden hesaplar
ve paylaşılan sonuçlarla karşılaştırır.
    PYTHONPATH=. python tools/verify_release.py
Kontroller:
  1. Faz 4 kirli/temiz/altın makro F1 (her veri seti × model × tohum) == release_data/results/model_results.csv
  2. IMDB klasik + transformer doğrulukları == release_data/results/imdb_tf_results.json"""
import json, subprocess, sys
import numpy as np, pandas as pd

ok = True
ref = pd.read_csv("release_data/results/model_results.csv")
subprocess.run([sys.executable, "scripts/run_eval_models.py"], check=True, capture_output=True)
new = pd.read_csv("out/model_results.csv")
m = ref.merge(new, on=["dataset", "seed", "model"], suffixes=("_ref", "_new"))
for c in ["dirty", "clean", "gold"]:
    d = float(np.abs(m[f"{c}_ref"] - m[f"{c}_new"]).max())
    print(f"Faz 4 {c:5s}: {len(m)} satır, en büyük fark {d:.2e}"); ok &= d < 1e-9 and len(m) == len(ref)

t = pd.read_parquet("out/imdb_preds.parquet")
r = json.load(open("release_data/results/imdb_tf_results.json"))["scores"]
for mdl in ["tfidf_svm", "tfidf_lr", "nb", "bert", "roberta"]:
    if mdl in ("bert", "roberta"):
        accs = []
        for s in range(3):
            p = t.h.map(pd.read_parquet(f"faz5_in/pred_{mdl}_s{s}.parquet").drop_duplicates("h").set_index("h").pred)
            accs.append((p.values == t.label.values).mean())
        a = float(np.mean(accs))
    else:
        a = float((t[f"pred_{mdl}"] == t.label).mean())
    d = abs(a - r[mdl]["dirty"]); print(f"IMDB {mdl:9s} kirli doğruluk {a:.5f} (fark {d:.1e})"); ok &= d < 1e-9
print("SONUÇ:", "TÜM KONTROLLER GEÇTİ" if ok else "UYUŞMAZLIK VAR"); sys.exit(0 if ok else 1)
