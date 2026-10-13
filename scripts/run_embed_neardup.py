"""Faz 2.2 gömme bileşeni: Colab çıktısı embed_pairs.csv.gz'yi yerel meta veriyle eşleştirip özetler.
Kullanım: python3 run_embed_neardup.py <embed_pairs.csv.gz>"""
import json, sys
import pandas as pd

pairs = pd.read_csv(sys.argv[1]).drop_duplicates(["dataset", "i", "j"])
meta = pd.read_parquet("out/embed_meta.parquet")
res = {}
for n, m in meta.groupby("dataset", sort=False):
    m = m.set_index("id")
    p = pairs[pairs.dataset == n]
    r = {"unique_texts": len(m), "pairs_cos>=0.90": len(p)}
    lab, spl = m["labels"], m["splits"]
    for t in (0.90, 0.95, 0.98):
        q = p[p.cos >= t]
        inv = set(q.i) | set(q.j)
        r[f"pct_texts_with_neighbor@{t}"] = round(100 * len(inv) / len(m), 3)
        r[f"label_conflict_pairs@{t}"] = int((lab.loc[q.i].values != lab.loc[q.j].values).sum())
        if (spl == "test").any():
            te_only = spl == "test"; tr_any = spl.str.contains("train")
            leak = set(q.i[te_only.loc[q.i].values & tr_any.loc[q.j].values]) | \
                   set(q.j[te_only.loc[q.j].values & tr_any.loc[q.i].values])
            r[f"test_neighbor_in_train_pct@{t}"] = round(100 * len(leak) / int(te_only.sum()), 3)
    res[n] = r
    ex = p[p.cos >= 0.95].sample(min(20, int((p.cos >= 0.95).sum())), random_state=0)
    pd.DataFrame({"a": m.text.loc[ex.i].values, "b": m.text.loc[ex.j].values, "cos": ex.cos.values,
                  "labels_a": lab.loc[ex.i].values, "labels_b": lab.loc[ex.j].values}
                 ).to_csv(f"out/embed_examples_{n}.csv", index=False)
json.dump(res, open("out/embed_neardup.json", "w"), indent=1)
print(json.dumps(res, indent=1))
