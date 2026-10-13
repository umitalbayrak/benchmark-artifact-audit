"""Faz 5.1 (devam): IMDB transformer (BERT-base, RoBERTa-base; 3 tohum) + sıfır-atış LLM sonuçlarının klasik modellerle birleştirilmesi.
Test türleri run_imdb.py ile aynı. Transformer skorları tohum ortalaması; bootstrap'te her örnekte tohum ortalaması alınır.
LLM yalnız alt örneklemde (rastgele 2.000 + incelenen 1.310) değerlendirildi: diğer modeller de aynı alt örneklemde raporlanır.
Çıktı: out/imdb_tf_results.json"""
import json, itertools
import numpy as np, pandas as pd
from scipy.stats import binomtest, kendalltau

t = pd.read_parquet("out/imdb_preds.parquet").reset_index(drop=True)
CL = ["tfidf_svm", "tfidf_lr", "nb"]; TF = ["bert", "roberta"]
P = {m: [t[f"pred_{m}"].values] for m in CL}
for m in TF:
    P[m] = []
    for s in range(3):
        p = pd.read_parquet(f"faz5_in/pred_{m}_s{s}.parquet").drop_duplicates("h").set_index("h").pred
        v = t.h.map(p); assert v.notna().all(), (m, s); P[m].append(v.values)
M = CL + TF
lab, labc = t.label.values, t.label_corr.values
cl = t.clean_drop_reason.eq("").values; co = t.label_corr.notna().values
def accs(arrs, ii):
    r = {"dirty": [], "clean": [], "corrected": [], "clean_corrected": []}
    for a in arrs:
        p = a[ii]
        r["dirty"].append(np.mean(p == lab[ii])); r["clean"].append(np.mean((p == lab[ii])[cl[ii]]))
        r["corrected"].append(np.mean((p == labc[ii])[co[ii]])); r["clean_corrected"].append(np.mean((p == labc[ii])[co[ii] & cl[ii]]))
    return {k: float(np.mean(v)) for k, v in r.items()}, {k: float(np.std(v)) for k, v in r.items()}
allix = np.arange(len(t)); out = {"scores": {}, "seed_std": {}}
for m in M:
    out["scores"][m], out["seed_std"][m] = accs(P[m], allix)
K = list(out["scores"]["bert"])
rng = np.random.default_rng(0); B = 2000
boot = {m: {k: [] for k in K} for m in M}
for _ in range(B):
    ii = rng.integers(0, len(t), len(t))
    for m in M:
        a, _ = accs(P[m], ii)
        for k in K: boot[m][k].append(a[k])
boot = {m: {k: np.array(v) for k, v in d.items()} for m, d in boot.items()}
ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
out["acc_ci"] = {m: {k: ci(v) for k, v in d.items()} for m, d in boot.items()}
out["corrected_minus_dirty"] = {m: [round(out["scores"][m]["corrected"] - out["scores"][m]["dirty"], 4), *ci(boot[m]["corrected"] - boot[m]["dirty"])] for m in M}
out["clean_minus_dirty"] = {m: [round(out["scores"][m]["clean"] - out["scores"][m]["dirty"], 4), *ci(boot[m]["clean"] - boot[m]["dirty"])] for m in M}
dv = [out["scores"][m]["dirty"] for m in M]; od = list(np.argsort(-np.array(dv)))
out["ranking"] = {k: sorted(M, key=lambda m: -out["scores"][m][k]) for k in K}
out["kendall"] = {k: float(kendalltau(dv, [out["scores"][m][k] for m in M]).statistic) for k in K}
out["boot_share_rank_equal_dirty"] = {k: float(np.mean([list(np.argsort(-np.array([boot[m][k][b] for m in M]))) == od for b in range(B)])) for k in K}
# Çift farkları: kirli vs düzeltilmiş (fark değişimi)
out["pair_gap"] = {}
for a, b in itertools.combinations(M, 2):
    g = {k: [round(out["scores"][a][k] - out["scores"][b][k], 4), *ci(boot[a][k] - boot[b][k])] for k in ["dirty", "corrected"]}
    g["gap_change"] = [round(g["corrected"][0] - g["dirty"][0], 4), *ci((boot[a]["corrected"] - boot[b]["corrected"]) - (boot[a]["dirty"] - boot[b]["dirty"]))]
    out["pair_gap"][f"{a}-{b}"] = g
# McNemar (tohum 0)
out["mcnemar_s0"] = {}
for a, b in itertools.combinations(M, 2):
    r = {}
    for k, y, msk in [("dirty", lab, np.ones(len(t), bool)), ("corrected", labc, co)]:
        ca, cb = P[a][0][msk] == y[msk], P[b][0][msk] == y[msk]; n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        r[k] = {"a_only": n01, "b_only": n10, "p": float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0}
    out["mcnemar_s0"][f"{a}-{b}"] = r
# İncelenen aday kümesi
rv = t.corr_status.isin(["flipped", "dropped_ambiguous", "reviewed_ok"]).values
fl = (t.corr_status == "flipped").values; amb = (t.corr_status == "dropped_ambiguous").values
out["reviewed_subset"] = {m: {"acc_vs_given": float(np.mean([np.mean(a[rv] == lab[rv]) for a in P[m]])),
                              "acc_vs_corrected": float(np.mean([np.mean((a == labc)[rv & co]) for a in P[m]])),
                              "flipped_agree_corrected": float(np.mean([np.mean(a[fl] == labc[fl]) for a in P[m]])),
                              "ambiguous_agree_given": float(np.mean([np.mean(a[amb] == lab[amb]) for a in P[m]]))} for m in M}
# LLM (alt örneklem)
llm = pd.read_parquet("faz5_in/llm.parquet").drop_duplicates("h")
out["llm_model"] = llm.model.iloc[0]
lp = t.h.map(llm.set_index("h").pred)
rnd = t.h.isin(llm.loc[llm.part == "random", "h"]).values
P2 = {**P, "llm": [lp.values]}
def sub_acc(arrs, msk, y, extra=None):
    m2 = msk & (extra if extra is not None else True)
    return float(np.mean([np.mean(a[m2] == y[m2]) for a in arrs]))
out["llm_subsets"] = {}
for m in M + ["llm"]:
    out["llm_subsets"][m] = {"random2000_dirty": sub_acc(P2[m], rnd, lab),
                             "random2000_corrected": sub_acc(P2[m], rnd & co, labc),
                             "reviewed_vs_given": sub_acc(P2[m], rv, lab),
                             "reviewed_vs_corrected": sub_acc(P2[m], rv & co, labc),
                             "flipped_agree_corrected": sub_acc(P2[m], fl, labc),
                             "ambiguous_agree_given": sub_acc(P2[m], amb, lab)}
# Yeniden ağırlıklandırılmış tahmini tam test (LLM için): rastgele 2000 örnek incelenmeyenleri temsil eder,
# incelenen 1.310 tamamı mevcut. Tam test doğruluğu ≈ (N_rest·acc_rnd + Σ incelenen)/N
N = len(t); Nrv = int(rv.sum()); Nco = int(co.sum()); Nrv_co = int((rv & co).sum())
for m in M + ["llm"]:
    a_r_d = out["llm_subsets"][m]["random2000_dirty"]; a_r_c = out["llm_subsets"][m]["random2000_corrected"]
    a_v_d = out["llm_subsets"][m]["reviewed_vs_given"]; a_v_c = out["llm_subsets"][m]["reviewed_vs_corrected"]
    out["llm_subsets"][m]["est_full_dirty"] = ((N - Nrv) * a_r_d + Nrv * a_v_d) / N
    out["llm_subsets"][m]["est_full_corrected"] = ((Nco - Nrv_co) * a_r_c + Nrv_co * a_v_c) / Nco
# LLM bootstrap: tahmini tam test, tabakalı (rastgele / incelenen)
ir, iv = np.where(rnd)[0], np.where(rv)[0]
lb = {m: {"d": [], "c": []} for m in M + ["llm"]}
for _ in range(B):
    a, b = rng.choice(ir, len(ir)), rng.choice(iv, len(iv))
    for m in M + ["llm"]:
        arrs = P2[m]
        rd = np.mean([np.mean(x[a] == lab[a]) for x in arrs]); vd = np.mean([np.mean(x[b] == lab[b]) for x in arrs])
        a2, b2 = a[co[a]], b[co[b]]
        rc = np.mean([np.mean(x[a2] == labc[a2]) for x in arrs]); vc = np.mean([np.mean(x[b2] == labc[b2]) for x in arrs])
        lb[m]["d"].append(((N - Nrv) * rd + Nrv * vd) / N); lb[m]["c"].append(((Nco - Nrv_co) * rc + Nrv_co * vc) / Nco)
lb = {m: {k: np.array(v) for k, v in d.items()} for m, d in lb.items()}
out["llm_est_ci"] = {m: {"dirty": ci(lb[m]["d"]), "corrected": ci(lb[m]["c"]), "gain": ci(lb[m]["c"] - lb[m]["d"])} for m in lb}
out["llm_vs"] = {m: {"dirty_diff": ci(lb["llm"]["d"] - lb[m]["d"]), "corrected_diff": ci(lb["llm"]["c"] - lb[m]["c"]),
                     "gap_change": ci((lb["llm"]["c"] - lb[m]["c"]) - (lb["llm"]["d"] - lb[m]["d"]))} for m in M}
M6 = M + ["llm"]
ed = [out["llm_subsets"][m]["est_full_dirty"] for m in M6]; ec = [out["llm_subsets"][m]["est_full_corrected"] for m in M6]
out["ranking_with_llm"] = {"dirty": sorted(M6, key=lambda m: -out["llm_subsets"][m]["est_full_dirty"]),
                           "corrected": sorted(M6, key=lambda m: -out["llm_subsets"][m]["est_full_corrected"]),
                           "tau": float(kendalltau(ed, ec).statistic)}
json.dump(out, open("out/imdb_tf_results.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
