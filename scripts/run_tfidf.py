"""Faz 4.2: TF-IDF + LR, aynı model üç testte (kirli / temiz / altın). Tahminler kaydedilir (4.5 istatistik için).
Ek: winvoker modeli TMRD test kümesinde (winvoker'da da geçen vs geçmeyen metinler) — çapraz kirlilik etkisi."""
import json, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from auditkit.models import tfidf_lr
from auditkit import core

def wf1(y, p, w, labels):
    cm = confusion_matrix(y, p, labels=labels, sample_weight=w).astype(float)
    f = []
    for i in range(len(labels)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)), cm

names = sys.argv[1:] or ["TMPR", "TMRD", "TRSAv1", "winvoker"]
res = json.load(open("out/tfidf_results.json")) if len(sys.argv) > 1 and __import__("os").path.exists("out/tfidf_results.json") else {}
for d in names:
    gold = pd.read_parquet(f"out/splits/gold_{d}.parquet")
    labels = sorted(gold.label.unique().tolist() + ([] if d in ("TMRD", "TMPR") else []))
    seeds = [0] if d == "winvoker" else [0, 1, 2]
    for s in seeds:
        t0 = time.time()
        x = pd.read_parquet(f"out/splits/{d}_s{s}.parquet")
        tr, te = x[x.part == "train"], x[x.part == "test"]
        labels = sorted(tr.label.unique())
        m = tfidf_lr().fit(tr.text, tr.label)
        pd_ = m.predict(te.text); pg = m.predict(gold.text)
        clean = (te.clean_drop_reason == "").values
        r = {"dirty_macroF1": f1_score(te.label, pd_, average="macro"), "dirty_acc": accuracy_score(te.label, pd_),
             "clean_macroF1": f1_score(te.label[clean], pd_[clean], average="macro"),
             "clean_acc": accuracy_score(te.label[clean], pd_[clean])}
        gl = [l for l in labels if l in set(gold.label)]
        r["gold_macroF1_w"], cm = wf1(gold.label, pg, gold.weight, labels)
        r["gold_acc_w"] = float(np.average(pg == gold.label.values, weights=gold.weight))
        r["gold_recall_w"] = {l: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i, l in enumerate(labels)}
        # Düşürülen satırlar üzerinde başarım (neden temiz test daha zor/kolay?)
        for why in ["exact_in_train", "neardup_in_train", "label_conflict"]:
            k = (te.clean_drop_reason == why).values
            if k.sum():
                r[f"acc_on_{why}"] = float(accuracy_score(te.label[k], pd_[k])); r[f"n_{why}"] = int(k.sum())
        r["seconds"] = round(time.time() - t0)
        res[f"{d}_s{s}"] = r
        pd.DataFrame({"pred": pd_}).to_parquet(f"out/splits/pred_tfidf_{d}_s{s}.parquet", index=False)
        pd.DataFrame({"pred": pg}).to_parquet(f"out/splits/predgold_tfidf_{d}_s{s}.parquet", index=False)
        print(d, s, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        if d == "winvoker":
            tm = core.add_keys(pd.read_parquet("out/splits/TMRD_s0.parquet").query("part == 'test'"))
            wk = set(core.add_keys(x[x.part == "train"][["text"]].assign(label="x")).k_fold)
            tm["in_winvoker"] = tm.k_fold.isin(wk) & (tm.n_words >= 4)
            p = m.predict(tm.text)
            bin_ = p != "neutral"                       # ikili test: nötr tahminleri ayrı raporlanır
            res["winvoker_on_TMRD"] = {g: {"n": int((tm.in_winvoker == g).sum()),
                                           "acc_excl_neutral_pred": float(accuracy_score(tm.label[(tm.in_winvoker == g) & bin_], p[(tm.in_winvoker == g) & bin_])),
                                           "neutral_pred_share": float((~bin_[tm.in_winvoker == g]).mean())}
                                       for g in (True, False)}
            print("winvoker->TMRD", json.dumps(res["winvoker_on_TMRD"], default=str), flush=True)
        json.dump(res, open("out/tfidf_results.json", "w"), indent=1, default=str)
