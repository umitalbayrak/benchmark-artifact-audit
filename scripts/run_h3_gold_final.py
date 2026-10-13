"""H3'ün insan doğrulamalı tekrarı: iki etiketleyicinin 'nötr' dediği TRSAv1 yorumlarında
winvoker modeli vs TRSAv1 modeli. Eğitim kümeleri doğrulama örneklerini içermez."""
import json
import numpy as np
import pandas as pd
from auditkit.loaders import load_winvoker, load_trsav1
from auditkit import core
from auditkit.models import tfidf_lr

k = pd.read_csv("out/annotation_final.csv")
t = k[k.dataset == "TRSAv1"].drop_duplicates("item_id")
gold = t[t.gold_final.isin(["neutral", "positive", "negative"])]
w = load_winvoker(); w = w[w.split == "train"]
tr = core.add_keys(load_trsav1()); tr = tr[~tr.k_norm.isin(set(t.k_norm))]   # sızıntıyı önle
mW = tfidf_lr().fit(w.text, w.label)
mT = tfidf_lr().fit(tr.text, tr.label)
res = {}
for name, m in (("winvoker_model", mW), ("trsav1_model", mT)):
    p = m.predict(gold.text)
    g = gold.gold_final.values
    res[name] = {c: {"n": int((g == c).sum()), "recall": round(float((p[g == c] == c).mean()), 3),
                     "pred_dist": pd.Series(p[g == c]).value_counts().to_dict()} for c in ("neutral", "positive", "negative")}
    res[name]["acc_all_gold"] = round(float((p == g).mean()), 3)
res["n_gold"] = len(gold)
json.dump(res, open("out/h3_gold_final.json", "w"), indent=1)
print(json.dumps(res, indent=1))
