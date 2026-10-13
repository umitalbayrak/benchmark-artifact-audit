"""Faz 1.5 envanter + Faz 2 ön ölçümleri (tam/normalize kopya, çakışma, sızıntı, örtüşme)."""
import json
import pandas as pd
from auditkit.loaders import LOADERS
from auditkit import core

pd.set_option("display.width", 200)
dfs = {n: core.add_keys(f()) for n, f in LOADERS.items()}

inv, audit = [], []
for n, d in dfs.items():
    inv.append({
        "dataset": n, "rows": len(d),
        "labels": json.dumps(d.label.value_counts().to_dict()),
        "splits": json.dumps(d.split.value_counts().to_dict()),
        "sources": json.dumps(d.source.value_counts().to_dict(), ensure_ascii=False),
        "unmapped_labels": int(d.label.isna().sum()),
        "empty_after_norm": int((d.k_norm == "").sum()),
        "median_words": float(d.n_words.median()),
        "pct_le3_words": round(100 * (d.n_words <= 3).mean(), 1),
    })
    row = {"dataset": n}
    for key in ("text", "k_norm"):
        s = core.duplicate_stats(d, key)
        row[f"{key}_redundant_pct"] = round(s["redundant_pct"], 2)
    s = core.duplicate_stats(d, "k_norm")
    row.update(conflict_texts=s["conflict_texts"], conflict_rows=s["conflict_rows"],
               conflict_rows_pct=round(s["conflict_rows_pct"], 2))
    if (d.split == "test").any():
        row.update({f"split_{k}": round(v, 2) for k, v in core.split_leakage(d).items()})
    row["cv_leaked_pct"] = round(core.cv_leakage(d)["cv_leaked_pct"], 2)
    audit.append(row)

inv, audit = pd.DataFrame(inv), pd.DataFrame(audit)
ov = core.cross_overlap(dfs)
inv.to_csv("out/inventory.csv", index=False)
audit.to_csv("out/audit_prelim.csv", index=False)
ov.to_csv("out/cross_overlap.csv", index=False)
print(inv.to_string()); print(); print(audit.T.to_string()); print(); print(ov[ov.shared_texts > 0].to_string())

# winvoker kaynak x etiket tablosu (H3 için)
w = dfs["winvoker"]
print(); print(pd.crosstab(w.source, w.label, margins=True).to_string())
