"""Faz 3.2: tabakalı örneklem + kör etiketleme dosyaları.

Popülasyon (prevalans tahmini için) = run_cleanlab.py'deki tekilleştirilmiş tablo
(boş normalize metin hariç, (k_norm, label) tekil). winvoker'da yalnızca yorum kaynakları
(wiki/random çıkarıldı; nötr sınıfın kaynağı zaten %100 belli).

Tabakalar:
  A  cleanlab adayı (flagged)                     -> prevalans: ağırlık N_A / n_A
  R  rastgele, cleanlab adayı olmayanlardan       -> prevalans: ağırlık N_R / n_R
  E  gömme cos >= 0,98, tek etiketli, etiketi çelişen çiftler (her iki metin) -> prevalansa girmez
  H  TRSAv1 nötr etiketli rastgele (H3 için)      -> prevalansa girmez
Çıktılar: out/annotation_key.csv (gizli anahtar), out/annotation_population.json (ağırlıklar),
share/Etiketleme_A.xlsx, share/Etiketleme_B.xlsx (aynı maddeler, farklı sıra).
"""
import json
import numpy as np
import pandas as pd
from auditkit.loaders import LOADERS
from auditkit import core

SEED = 20261007
PLAN = {  # dataset: (n_A, n_R, n_E_pairs, n_H)
    "TRSAv1": (100, 120, 30, 60),
    "winvoker": (100, 120, 20, 0),
    "TMRD": (80, 120, 0, 0),
    "TMPR": (80, 120, 0, 0),
}
rng = np.random.default_rng(SEED)
items, pop = [], {}
emb_pairs = pd.read_csv("out/embed_pairs.csv.gz").drop_duplicates(["dataset", "i", "j"])
emb_meta = pd.read_parquet("out/embed_meta.parquet")

for n, (nA, nR, nE, nH) in PLAN.items():
    d = core.add_keys(LOADERS[n]())
    d = d[d.k_norm != ""].drop_duplicates(["k_norm", "label"]).reset_index(drop=True)
    cand = pd.read_csv(f"out/cleanlab_candidates_{n}.csv")[["text", "label"]].assign(flagged=True)
    d = d.merge(cand.drop_duplicates(), on=["text", "label"], how="left")
    d["flagged"] = d.flagged.fillna(False).astype(bool)
    if n == "winvoker":
        d = d[~d.source.isin(["wiki", "random"])].reset_index(drop=True)
    NA, NR = int(d.flagged.sum()), int((~d.flagged).sum())
    pop[n] = {"N_A": NA, "N_R": NR, "n_A": nA, "n_R": nR}
    A = d[d.flagged].sample(nA, random_state=SEED)
    R = d[~d.flagged].sample(nR, random_state=SEED)
    for s, g in (("A", A), ("R", R)):
        for _, r in g.iterrows():
            items.append(dict(dataset=n, stratum=s, text=r.text, k_norm=r.k_norm, orig_label=r.label,
                              source=r.source, pair_id=""))
    used = set(A.k_norm) | set(R.k_norm)
    if nE:
        m = emb_meta[emb_meta.dataset == n].set_index("id")
        p = emb_pairs[(emb_pairs.dataset == n) & (emb_pairs.cos >= 0.98)]
        li, lj = m.labels.loc[p.i].values, m.labels.loc[p.j].values
        ok = (li != lj) & ~pd.Series(li).str.contains("|", regex=False).values \
             & ~pd.Series(lj).str.contains("|", regex=False).values
        p = p[ok]
        p = p.sample(min(nE, len(p)), random_state=SEED)
        for k, (i, j, c) in enumerate(zip(p.i, p.j, p.cos)):
            for idx in (i, j):
                items.append(dict(dataset=n, stratum="E", text=m.text.loc[idx], k_norm=m.k_norm.loc[idx],
                                  orig_label=m.labels.loc[idx], source="", pair_id=f"{n}-E{k:03d}"))
        pop[n]["E_pairs"] = int(len(p)); pop[n]["E_pairs_available"] = int(ok.sum())
    if nH:
        H = d[(d.label == "neutral") & ~d.k_norm.isin(used)].sample(nH, random_state=SEED + 1)
        for _, r in H.iterrows():
            items.append(dict(dataset=n, stratum="H", text=r.text, k_norm=r.k_norm, orig_label=r.label,
                              source=r.source, pair_id=""))

key = pd.DataFrame(items)
# Aynı metin birden fazla tabakaya düştüyse (ör. E ile A), etiketleyiciye bir kez gösterilir
key["show_id"] = key.groupby(["dataset", "k_norm"]).ngroup()
shown = key.drop_duplicates("show_id")[["show_id", "text"]].sample(frac=1, random_state=SEED).reset_index(drop=True)
shown["item_id"] = [f"M{k+1:04d}" for k in range(len(shown))]
key = key.merge(shown[["show_id", "item_id"]], on="show_id")
key.to_csv("out/annotation_key.csv", index=False)
json.dump(pop, open("out/annotation_population.json", "w"), indent=1)
print(key.groupby(["dataset", "stratum"]).size().unstack(fill_value=0))
print("toplam madde (etiketleyiciye gösterilen):", len(shown))
shown[["item_id", "text"]].to_csv("out/annotation_items.csv", index=False)
print(json.dumps(pop, indent=1))
