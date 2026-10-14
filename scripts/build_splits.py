"""Faz 4.1: kirli / temiz / altın test tanımları ve bölümlerin üretimi.

Protokol (veri seti d, tohum s):
  0. İnsan doğrulamasına giren tüm metinler (987 madde, normalize anahtar) veri setinden önce çıkarılır;
     böylece hiçbir modelin eğitimine girmez.
  1. Bölüm S: winvoker'da resmî train/test; diğerlerinde satır düzeyinde tabakalı rastgele %80/%20
     (literatürdeki yaygın uygulama; kopyalar olduğu gibi kalır). Tohumlar 0, 1, 2.
  2. Model train(S) ile eğitilir (4.2 / 4.3).
  3. Aynı model üç test kümesinde değerlendirilir:
     T_dirty = test(S), olduğu gibi.
     T_clean = test(S) eksi şu satırlar:
        (a) normalize metni train(S)'de de geçenler (birebir/normalize sızıntı),
        (b) train(S)'deki bir metinle yakın/anlamsal kopyası olanlar (MinHash Jaccard >= 0,8 veya gömme cos >= 0,98),
        (c) veri setinde farklı etiketlerle geçen normalize metinler (etiket çakışması),
        (d) yalnız emoji/noktalamadan oluşan metinler.
     T_gold = insan doğrulamalı A ve R tabakaları (altın etiket pozitif/negatif/nötr; ikili setlerde nötr hariç),
              tabaka ağırlıklarıyla (N_A, N_R) ağırlıklandırılır.
Çıktı: out/splits/<d>_s<seed>.parquet (text, label, part, k_norm, clean_drop_reason) ve out/splits/gold_<d>.parquet
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from auditkit.loaders import LOADERS
from auditkit import core
from auditkit.neardup import near_duplicate_pairs

os.makedirs("out/splits", exist_ok=True)
fin = pd.read_csv("out/annotation_final.csv")
pop = json.load(open("out/annotation_population.json"))
emb = pd.read_csv("out/embed_pairs.csv.gz").drop_duplicates(["dataset", "i", "j"])
emb = emb[emb.cos >= 0.98]
meta = pd.read_parquet("out/embed_meta.parquet")
summary = {}

for d in ["TMPR", "TMRD", "TRSAv1", "winvoker"]:
    df = core.add_keys(LOADERS[d]())
    annotated = set(fin.loc[fin.dataset == d, "k_norm"])
    df = df[~df.k_norm.isin(annotated)].reset_index(drop=True)
    conflict = set(df.groupby("k_norm").label.nunique().loc[lambda s: s > 1].index)

    # Yakın/anlamsal kopya grafiği: normalize metin -> komşu normalize metinler
    nbr = {}
    m = meta[meta.dataset == d].set_index("id").k_norm
    for i, j in zip(emb[emb.dataset == d].i, emb[emb.dataset == d].j):
        a, b = m.loc[i], m.loc[j]
        nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)
    u = sorted(set(df.loc[df.n_words >= 5, "k_norm"]))
    for _, r in near_duplicate_pairs(u, threshold=0.8).iterrows():
        a, b = u[int(r.i)], u[int(r.j)]
        nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)

    seeds = [0] if d == "winvoker" else [0, 1, 2]
    for s in seeds:
        x = df.copy()
        if d == "winvoker":
            x["part"] = x.split
        else:
            tr, te = train_test_split(x.index, test_size=0.2, stratify=x.label, random_state=s)
            x["part"] = "train"; x.loc[te, "part"] = "test"
        train_keys = set(x.loc[x.part == "train", "k_norm"])
        reason = pd.Series("", index=x.index)
        te = x.part == "test"
        reason[te & x.k_norm.eq("")] = "empty"
        reason[te & reason.eq("") & x.k_norm.isin(conflict)] = "label_conflict"
        reason[te & reason.eq("") & x.k_norm.isin(train_keys)] = "exact_in_train"
        near = x.k_norm.map(lambda k: bool(nbr.get(k, set()) & train_keys))
        reason[te & reason.eq("") & near] = "neardup_in_train"
        x["clean_drop_reason"] = reason
        x[["text", "label", "part", "k_norm", "source", "clean_drop_reason"]].to_parquet(f"out/splits/{d}_s{s}.parquet", index=False)
        t = x[te]
        summary[f"{d}_s{s}"] = {"train": int((x.part == "train").sum()), "test_dirty": int(te.sum()),
                                "test_clean": int((te & reason.eq("")).sum()),
                                "dropped": t.clean_drop_reason[t.clean_drop_reason != ""].value_counts().to_dict()}

    g = fin[(fin.dataset == d) & fin.stratum.isin(["A", "R"]) & fin.gold_final.isin(["positive", "negative", "neutral"])]
    if d in ("TMRD", "TMPR"):
        g = g[g.gold_final != "neutral"]
    g = g.drop_duplicates(["item_id"])
    P = pop[d]
    nA, nR = (g.stratum == "A").sum(), (g.stratum == "R").sum()
    # Tasarım (içerilme) ağırlığı: N / çekilen örnek sayısı (değerlendirilemez maddeler çıkarılmadan önceki n)
    g = g.assign(weight=np.where(g.stratum == "A", P["N_A"] / P["n_A"], P["N_R"] / P["n_R"]))
    g[["text", "gold_final", "orig_label", "stratum", "weight"]].rename(columns={"gold_final": "label"}) \
        .to_parquet(f"out/splits/gold_{d}.parquet", index=False)
    summary[f"gold_{d}"] = {"n": len(g), "A": int(nA), "R": int(nR)}
    print(d, json.dumps({k: v for k, v in summary.items() if k.startswith(d) or k == f"gold_{d}"}, ensure_ascii=False), flush=True)
json.dump(summary, open("out/splits/summary.json", "w"), indent=1)
