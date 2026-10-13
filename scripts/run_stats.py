"""Faz 4.4 değerlendirme + 4.5 istatistik.
- LLM (Qwen3-8B, sıfır-atış): kirli (2.000 örneklem), temiz (örneklemin temiz kısmı), altın (ağırlıklı).
- Altın makro F1 için tabakalı bootstrap %95 GA (tohum ortalamalı), model çiftleri için eşleştirilmiş bootstrap fark GA.
- Kendall τ: kirli sıralama vs altın sıralama; bootstrap'te altın sıralamanın kirli sıralamayla aynı kalma oranı.
- McNemar (kesin, binom): model çiftleri, kirli ve temiz testte (tohum 0) ve altın kümede.
- Çapraz kirlilik: winvoker modeli TMRD'de, uzunluk çeyreklerine göre katmanlı karşılaştırma."""
import json, itertools
import numpy as np, pandas as pd
from scipy.stats import binomtest, kendalltau
from sklearn.metrics import f1_score, confusion_matrix

rng = np.random.default_rng(0)
B = 2000
DS = ["TMPR", "TMRD", "TRSAv1", "winvoker"]
MAIN = ["tfidf", "berturk", "xlmr"]

def wf1(y, p, w, labels):
    cm = confusion_matrix(y, p, labels=labels, sample_weight=w).astype(float); f = []
    for i in range(len(labels)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

def preds(model, d, s, gold=False):
    pre = "predgold" if gold else "pred"
    if model in ("berturk", "xlmr"):
        return pd.read_parquet(f"faz4_in/{pre}_{model}_{d}_s{s}.parquet").pred.values
    if model == "llm":
        part = "gold" if gold else "test"
        return pd.read_parquet(f"faz4_llm/{d}_{part}.parquet")
    tag = "tfidf100k" if d == "winvoker" else "tfidf"
    return pd.read_parquet(f"out/splits/{pre}_{tag}_{d}_s{s}.parquet").pred.values

out = {"llm": {}, "gold_ci": {}, "gold_pair_diff": {}, "kendall": {}, "mcnemar": {}}
for d in DS:
    gold = pd.read_parquet(f"out/splits/gold_{d}.parquet").reset_index(drop=True)
    x0 = pd.read_parquet(f"out/splits/{d}_s0.parquet"); labels = sorted(x0.label.unique())
    te0 = x0[x0.part == "test"]
    # --- LLM
    lt = preds("llm", d, 0).set_index("idx"); lg = preds("llm", d, 0, gold=True).set_index("idx")
    smp = te0.loc[lt.index]; cl = (smp.clean_drop_reason == "").values
    out["llm"][d] = {"model": lt.model.iloc[0], "n_test_sample": len(smp),
                     "dirty": f1_score(smp.label, lt.pred, average="macro"),
                     "clean": f1_score(smp.label[cl], lt.pred.values[cl], average="macro"),
                     "gold": wf1(gold.label, lg.loc[gold.index].pred, gold.weight, labels)}
    # --- altın tahminleri: model -> [tohum dizileri]
    P = {m: [preds(m, d, s, gold=True) for s in range(3)] for m in MAIN}
    P["llm"] = [lg.loc[gold.index].pred.values]
    idx_by = {st: np.where(gold.stratum.values == st)[0] for st in ("A", "R")}
    boot = {m: [] for m in P}
    for _ in range(B):
        ii = np.concatenate([rng.choice(v, len(v)) for v in idx_by.values()])
        y, w = gold.label.values[ii], gold.weight.values[ii]
        for m, arrs in P.items():
            boot[m].append(np.mean([wf1(y, a[ii], w, labels) for a in arrs]))
    boot = {m: np.array(v) for m, v in boot.items()}
    point = {m: np.mean([wf1(gold.label, a, gold.weight, labels) for a in arrs]) for m, arrs in P.items()}
    out["gold_ci"][d] = {m: [round(point[m], 4), round(float(np.percentile(boot[m], 2.5)), 4), round(float(np.percentile(boot[m], 97.5)), 4)] for m in P}
    out["gold_pair_diff"][d] = {}
    for a, b in itertools.combinations(list(P), 2):
        diff = boot[a] - boot[b]
        out["gold_pair_diff"][d][f"{a}-{b}"] = [round(point[a] - point[b], 4), round(float(np.percentile(diff, 2.5)), 4),
                                                round(float(np.percentile(diff, 97.5)), 4)]
    # --- Kendall τ (ana 3 model): kirli ortalama vs altın
    res = pd.read_csv("out/model_results.csv"); res = res[res.dataset == d].groupby("model")[["dirty", "clean", "gold"]].mean()
    dirty_rank = res.dirty.loc[MAIN].values
    tau_point = kendalltau(dirty_rank, [point[m] for m in MAIN]).statistic
    tau_clean = kendalltau(dirty_rank, res.clean.loc[MAIN].values).statistic
    order_d = list(np.argsort(-dirty_rank))
    same = np.mean([list(np.argsort(-np.array([boot[m][k] for m in MAIN]))) == order_d for k in range(B)])
    taus = [kendalltau(dirty_rank, [boot[m][k] for m in MAIN]).statistic for k in range(B)]
    out["kendall"][d] = {"tau_dirty_vs_clean": float(tau_clean), "tau_dirty_vs_gold": float(tau_point),
                         "boot_share_gold_rank_equals_dirty": float(same),
                         "boot_tau_ci": [float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))]}
    # --- McNemar (tohum 0), kirli / temiz / altın
    te_p = {m: preds(m, d, 0) for m in MAIN}; cl0 = (te0.clean_drop_reason == "").values
    out["mcnemar"][d] = {}
    for a, b in itertools.combinations(MAIN, 2):
        r = {}
        for nm, y, pa, pb in [("dirty", te0.label.values, te_p[a], te_p[b]),
                              ("clean", te0.label.values[cl0], te_p[a][cl0], te_p[b][cl0]),
                              ("gold", gold.label.values, P[a][0], P[b][0])]:
            ca, cb = pa == y, pb == y; n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            r[nm] = {"a_only": n01, "b_only": n10, "p": float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}
        out["mcnemar"][d][f"{a}-{b}"] = r
    print(d, "done", flush=True)

# --- Çapraz kirlilik: uzunluk çeyreklerine göre
from auditkit import core
from auditkit.loaders import load_winvoker
tm = core.add_keys(pd.read_parquet("out/splits/TMRD_s0.parquet").query("part == 'test'"))
wv = core.add_keys(load_winvoker().query("split == 'train'"))
tm["in_wv"] = tm.k_fold.isin(set(wv.loc[wv.n_words >= 4, "k_fold"])) & (tm.n_words >= 4)
json_tf = json.load(open("out/tfidf_results.json"))
out["cross_contam_note"] = "uzunluk kontrolü için winvoker modeli tahminleri yeniden üretilmeli; bkz. run_cross_len.py"
tm[["text", "label", "n_words", "in_wv"]].to_parquet("out/tmrd_test_overlap.parquet", index=False)

json.dump(out, open("out/stats_4_5.json", "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ["llm", "gold_ci", "kendall"]}, indent=1, default=float))
print(json.dumps(out["gold_pair_diff"], indent=1))
