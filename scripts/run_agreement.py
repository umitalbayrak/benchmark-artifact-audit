"""Faz 3.4: etiketleyici uyumu, uzlaşma listesi, ağırlıklı etiket hatası oranı."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

L = {"Pozitif": "positive", "Negatif": "negative", "Nötr": "neutral",
     "Belirsiz": "ambiguous", "Konu dışı / anlamsız": "offtopic"}
key = pd.read_csv("out/annotation_key.csv")
items = pd.read_csv("out/annotation_items.csv")
ann = {}
for f in "AB":
    x = pd.read_excel(f"annot_in/{f}.xlsx", sheet_name="Etiketleme")
    x.columns = ["item_id", "text", "label", "note"]
    chk = x.merge(items, on="item_id", suffixes=("", "_orig"))
    assert len(chk) == len(items) and (chk.text.astype(str) == chk.text_orig.astype(str)).all(), f"{f}: metin değişmiş"
    ann[f] = x.set_index("item_id").label.map(L)
lab = pd.DataFrame({"A": ann["A"], "B": ann["B"]})
out = {"missing_A": int(lab.A.isna().sum()), "missing_B": int(lab.B.isna().sum())}
both = lab.dropna()
out["n_both"] = len(both)
out["kappa_5class"] = cohen_kappa_score(both.A, both.B)
out["raw_agreement_5class"] = float((both.A == both.B).mean())
pol = both[both.A.isin(["positive", "negative", "neutral"]) & both.B.isin(["positive", "negative", "neutral"])]
out["kappa_3class_both_polar"] = cohen_kappa_score(pol.A, pol.B)
out["confusion"] = pd.crosstab(both.A, both.B).to_dict()

k = key.merge(lab, left_on="item_id", right_index=True)
k["agree"] = k.A.notna() & (k.A == k.B)
k["gold"] = k.A.where(k.agree)          # uyuşmazlık / eksik -> NaN
per = {}
for d, g in k.drop_duplicates(["item_id", "dataset"]).groupby("dataset"):
    gg = g.dropna(subset=["A", "B"])
    per[d] = {"n": len(gg), "kappa": round(cohen_kappa_score(gg.A, gg.B), 3),
              "agree": round(float((gg.A == gg.B).mean()), 3)}
out["per_dataset"] = per


def is_error(r):
    """Uzlaşılmış altın etikete göre orijinal etiket hatalı mı. None = değerlendirilemez."""
    if pd.isna(r.gold):
        return None
    if r.gold in ("ambiguous", "offtopic"):
        return "unratable"
    if r.gold == "neutral" and r.dataset in ("TMRD", "TMPR"):
        return "neutral_in_binary"
    return "error" if r.gold != r.orig_label else "correct"


k["verdict"] = k.apply(is_error, axis=1)
pop = json.load(open("out/annotation_population.json"))
rng = np.random.default_rng(0)
prev = {}
for d, P in pop.items():
    row = {}
    s = {}
    for st in ("A", "R"):
        g = k[(k.dataset == d) & (k.stratum == st)]
        v = g.verdict.value_counts().to_dict()
        rated = g[g.verdict.isin(["error", "correct"])]
        s[st] = rated.verdict.eq("error").to_numpy()
        row[f"{st}_counts"] = {kk: int(vv) for kk, vv in v.items()}
        row[f"{st}_n_rated"] = len(rated)
        row[f"{st}_error_rate"] = round(float(s[st].mean()), 4) if len(rated) else None
    NA, NR = P["N_A"], P["N_R"]
    w = lambda a, r: (NA * a.mean() + NR * r.mean()) / (NA + NR)
    est = w(s["A"], s["R"])
    boots = [w(rng.choice(s["A"], len(s["A"])), rng.choice(s["R"], len(s["R"]))) for _ in range(5000)]
    row["weighted_error_rate"] = round(float(est), 4)
    row["ci95"] = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
    row["cleanlab_precision_A"] = row["A_error_rate"]
    prev[d] = row
out["prevalence"] = prev

# E çiftleri: insanlar iki parafrazı aynı etiketledi mi? (aynıysa orijinal etiketlerden biri kesin hatalı)
e = k[k.stratum == "E"].dropna(subset=["pair_id"])
ep = e.groupby("pair_id").agg(gold=("gold", lambda s: tuple(s)), ds=("dataset", "first"))
ep["both_gold"] = ep.gold.map(lambda t: len(t) == 2 and not any(pd.isna(x) for x in t))
ep["same_gold"] = ep.gold.map(lambda t: len(t) == 2 and not any(pd.isna(x) for x in t) and t[0] == t[1])
out["E_pairs"] = {d: {"pairs": int(len(g)), "both_agreed": int(g.both_gold.sum()),
                      "humans_same_label": int(g.same_gold.sum())} for d, g in ep.groupby("ds")}

k.to_csv("out/annotation_merged.csv", index=False)
dis = k[~k.agree].drop_duplicates("item_id")[["item_id", "text", "A", "B"]]
dis.to_csv("out/disagreements.csv", index=False)
json.dump(out, open("out/agreement.json", "w"), indent=1, default=str, ensure_ascii=False)
print(json.dumps({kk: vv for kk, vv in out.items() if kk != "confusion"}, indent=1, default=str, ensure_ascii=False))
print(pd.crosstab(both.A, both.B, margins=True))
print("uyuşmazlık:", len(dis))
