"""Faz 3.4 (kesin): uzlaşılmış altın etiket = iki etiketleyici aynıysa o, değilse uzlaşma kararı.
Kesin ağırlıklı etiket hatası oranları, E çiftleri, H tabakası özeti. κ ilk bağımsız etiketlerden (run_agreement.py)."""
import json
import numpy as np
import pandas as pd

L = {"Pozitif": "positive", "Negatif": "negative", "Nötr": "neutral",
     "Belirsiz": "ambiguous", "Konu dışı / anlamsız": "offtopic"}
k = pd.read_csv("out/annotation_merged.csv")
u = pd.read_excel("annot_in/Uzlasma.xlsx", sheet_name="Uzlaşma")
u = u.rename(columns={"No": "item_id", "Uzlaşılan etiket": "adj"})[["item_id", "adj"]]
u["adj"] = u.adj.map(L)
assert u.adj.notna().all() and u.item_id.is_unique
k = k.merge(u, on="item_id", how="left")
k["gold_final"] = k.gold.where(k.agree, k.adj)
assert k.gold_final.notna().all()
k["source_of_gold"] = np.where(k.agree, "agreed", "adjudicated")

# Uzlaşma kararının ilk etiketlerden birine eşit olup olmadığı (kontrol)
dis = k[~k.agree].drop_duplicates("item_id")
adj_in_pair = (dis.adj == dis.A) | (dis.adj == dis.B)
out = {"adjudicated_items": int(len(dis)), "adj_equals_one_annotator": int(adj_in_pair.sum()),
       "adj_label_dist": dis.adj.value_counts().to_dict()}


def verdict(r):
    g = r.gold_final
    if g in ("ambiguous", "offtopic"):
        return "unratable"
    if g == "neutral" and r.dataset in ("TMRD", "TMPR"):
        return "neutral_in_binary"
    return "error" if g != r.orig_label else "correct"


k["verdict"] = k.apply(verdict, axis=1)
pop = json.load(open("out/annotation_population.json"))
rng = np.random.default_rng(0)
prev = {}
for d, P in pop.items():
    row, s = {}, {}
    for st in ("A", "R"):
        g = k[(k.dataset == d) & (k.stratum == st)]
        row[f"{st}_counts"] = {a: int(b) for a, b in g.verdict.value_counts().items()}
        rated = g[g.verdict.isin(["error", "correct"])]
        s[st] = rated.verdict.eq("error").to_numpy()
        row[f"{st}_n_rated"] = len(rated)
        row[f"{st}_error_rate"] = round(float(s[st].mean()), 4)
    NA, NR = P["N_A"], P["N_R"]
    w = lambda a, r: (NA * a.mean() + NR * r.mean()) / (NA + NR)
    boots = [w(rng.choice(s["A"], len(s["A"])), rng.choice(s["R"], len(s["R"]))) for _ in range(5000)]
    row["weighted_error_rate"] = round(float(w(s["A"], s["R"])), 4)
    row["ci95"] = [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)]
    # Hataların türü (rastgele tabaka): hangi orijinal etiket hangi altın etikete
    r = k[(k.dataset == d) & (k.stratum.isin(["A", "R"])) & (k.verdict == "error")]
    row["error_types_AR"] = (r.orig_label + "->" + r.gold_final).value_counts().to_dict()
    # "Değerlendirilemez" payı (belirsiz/konu dışı + ikili setlerde nötr), ağırlıklı
    un = {st: k[(k.dataset == d) & (k.stratum == st)].verdict.isin(["unratable", "neutral_in_binary"]).mean()
          for st in ("A", "R")}
    row["weighted_unratable_share"] = round(float((NA * un["A"] + NR * un["R"]) / (NA + NR)), 4)
    prev[d] = row
out["prevalence"] = prev

e = k[k.stratum == "E"]
ep = e.groupby("pair_id").agg(g=("gold_final", list), ds=("dataset", "first"))
ep["same"] = ep.g.map(lambda t: len(t) == 2 and t[0] == t[1])
out["E_pairs"] = {d: {"pairs": int(len(g)), "humans_same_label": int(g.same.sum())} for d, g in ep.groupby("ds")}
h = k[(k.dataset == "TRSAv1") & (k.stratum == "H")].drop_duplicates("item_id")
out["H_trsav1_neutral_labeled_gold"] = h.gold_final.value_counts().to_dict()
k.to_csv("out/annotation_final.csv", index=False)
json.dump(out, open("out/final_gold.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
