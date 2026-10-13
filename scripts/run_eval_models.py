"""Faz 4.3 değerlendirme: tüm modeller × veri setleri × tohumlar için kirli / temiz / altın makro F1.
Model sıralaması (3 model: TF-IDF, BERTurk, XLM-R) her test türünde; winvoker'da TF-IDF 100 bin kontrolü kullanılır."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

def wf1(y, p, w, labels):
    cm = confusion_matrix(y, p, labels=labels, sample_weight=w).astype(float); f = []
    for i in range(len(labels)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), {l: (cm[i, i] / cm[i].sum() if cm[i].sum() else np.nan) for i, l in enumerate(labels)}

def preds(model, d, s, gold=False):
    pre = "predgold" if gold else "pred"
    if model in ("berturk", "xlmr"):
        return pd.read_parquet(f"faz4_in/{pre}_{model}_{d}_s{s}.parquet").pred.values
    tag = "tfidf100k" if (model == "tfidf" and d == "winvoker") else "tfidf"
    return pd.read_parquet(f"out/splits/{pre}_{tag}_{d}_s{s}.parquet").pred.values

rows = []
for d in ["TMPR", "TMRD", "TRSAv1", "winvoker"]:
    gold = pd.read_parquet(f"out/splits/gold_{d}.parquet")
    for s in [0, 1, 2]:
        x = pd.read_parquet(f"out/splits/{d}_s{0 if d == 'winvoker' else s}.parquet")
        te = x[x.part == "test"]; clean = (te.clean_drop_reason == "").values
        labels = sorted(x.label.unique())
        for m in ["tfidf", "berturk", "xlmr"]:
            p = preds(m, d, s); pg = preds(m, d, s, gold=True)
            assert len(p) == len(te) and len(pg) == len(gold)
            gf, rec = wf1(gold.label, pg, gold.weight, labels)
            rows.append(dict(dataset=d, seed=s, model=m,
                             dirty=f1_score(te.label, p, average="macro"),
                             clean=f1_score(te.label[clean], p[clean], average="macro"),
                             gold=gf, **{f"gold_rec_{k}": v for k, v in rec.items()}))
r = pd.DataFrame(rows)
r.to_csv("out/model_results.csv", index=False)
agg = r.groupby(["dataset", "model"])[["dirty", "clean", "gold"]].agg(["mean", "std"]).round(4)
print(agg.to_string())
rec = r.groupby(["dataset", "model"])[[c for c in r.columns if c.startswith("gold_rec")]].mean().round(3)
print(rec.to_string())
# Sıralamalar (ortalama skorla)
mean = r.groupby(["dataset", "model"])[["dirty", "clean", "gold"]].mean()
rank = {d: {t: mean.loc[d][t].sort_values(ascending=False).index.tolist() for t in ["dirty", "clean", "gold"]}
        for d in mean.index.get_level_values(0).unique()}
print(json.dumps(rank, indent=1))
json.dump(rank, open("out/model_rankings.json", "w"), indent=1)
