"""winvoker TF-IDF, transformer koşullarıyla aynı 100 bin tabakalı eğitim alt örneklemi (tohum 0,1,2)."""
import json, numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from auditkit.models import tfidf_lr
from sklearn.metrics import confusion_matrix
def wf1(y, p, w, labels):
    cm = confusion_matrix(y, p, labels=labels, sample_weight=w).astype(float); f = []
    for i in range(len(labels)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), cm
x = pd.read_parquet("out/splits/winvoker_s0.parquet"); gold = pd.read_parquet("out/splits/gold_winvoker.parquet")
tr0, te = x[x.part == "train"], x[x.part == "test"]; clean = (te.clean_drop_reason == "").values
res = {}
for s in [0, 1, 2]:
    tr = tr0.groupby("label").sample(frac=100_000 / len(tr0), random_state=s)
    m = tfidf_lr().fit(tr.text, tr.label); labels = sorted(tr.label.unique())
    p = m.predict(te.text); pg = m.predict(gold.text)
    f, cm = wf1(gold.label, pg, gold.weight, labels)
    res[f"s{s}"] = {"n_train": len(tr), "dirty_macroF1": f1_score(te.label, p, average="macro"),
                    "clean_macroF1": f1_score(te.label[clean], p[clean], average="macro"), "gold_macroF1_w": f}
    pd.DataFrame({"pred": p}).to_parquet(f"out/splits/pred_tfidf100k_winvoker_s{s}.parquet", index=False)
    pd.DataFrame({"pred": pg}).to_parquet(f"out/splits/predgold_tfidf100k_winvoker_s{s}.parquet", index=False)
    print(s, res[f"s{s}"], flush=True)
json.dump(res, open("out/tfidf_wv100k.json", "w"), indent=1)
