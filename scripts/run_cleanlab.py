"""Faz 2.7: cleanlab ile şüpheli etiket adayları.

Her veri setinde 5 katlı çapraz doğrulamayla dışarıda-tutulmuş (out-of-fold) olasılıklar üretilir,
cleanlab.find_label_issues ile şüpheli örnekler sıralanır. IMDB'de Northcutt vd. (2021) MTurk
doğrulamasıyla karşılaştırılır (yöntemin doğrulaması).
Bellek için kopyalar önce normalize metne göre tekilleştirilir (aynı metin tek temsilciyle).
"""
import json, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from cleanlab.filter import find_label_issues
from auditkit.loaders import LOADERS, LOADERS_EN
from auditkit import core
from auditkit.models import tfidf_lr

names = sys.argv[1:] or ["TMPR", "TMRD", "TRSAv1", "IMDB", "winvoker"]
allL = {**LOADERS, **LOADERS_EN}
summary = {}
for n in names:
    t0 = time.time()
    d = core.add_keys(allL[n]())
    if n == "IMDB":
        d = d[d.split == "test"]                       # Northcutt test kümesini denetledi
    d = d[d.k_norm != ""]
    # aynı normalize metin + aynı etiket tek satır; çakışan etiketler ayrı satır kalır
    d = d.drop_duplicates(["k_norm", "label"]).reset_index(drop=True)
    classes = sorted(d.label.unique())
    y = d.label.map({c: i for i, c in enumerate(classes)}).values
    probs = np.zeros((len(d), len(classes)), dtype=np.float32)
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(d, y)):
        m = tfidf_lr(n_features=2**18).fit(d.text.iloc[tr], y[tr])
        probs[te] = m.predict_proba(d.text.iloc[te])
    issues = find_label_issues(y, probs, return_indices_ranked_by="self_confidence")
    d["oof_pred"] = [classes[i] for i in probs.argmax(1)]
    d["self_conf"] = probs[np.arange(len(d)), y]
    d["flagged"] = False
    d.loc[issues, "flagged"] = True
    res = {"rows_audited": len(d), "flagged": int(d.flagged.sum()),
           "flagged_pct": 100 * d.flagged.mean(),
           "oof_acc": float((d.oof_pred == d.label).mean()), "seconds": round(time.time() - t0, 1)}
    if n == "IMDB":
        conf = set(d.doc_id[d.mturk_status == "error"]); rev = set(d.doc_id[d.mturk_status.notna()])
        fl = set(d.doc_id[d.flagged])
        res["northcutt_confirmed_errors"] = len(conf)
        res["recall_of_northcutt_errors"] = len(fl & conf) / max(len(conf), 1)
        res["precision_among_reviewed"] = len(fl & conf) / max(len(fl & rev), 1)
    summary[n] = res
    cols = [c for c in ["text", "label", "oof_pred", "self_conf", "source", "split", "doc_id", "mturk_status"] if c in d]
    d.loc[issues, cols].to_csv(f"out/cleanlab_candidates_{n}.csv", index=False)
    print(n, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    json.dump(summary, open(f"out/cleanlab_{'_'.join(names)}.json", "w"), indent=1)
