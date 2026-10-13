"""Faz 2.2: yakın kopya denetimi, beş veri seti."""
import json, sys, time
import pandas as pd
from auditkit.loaders import LOADERS, LOADERS_EN
from auditkit import core
from auditkit.neardup import near_dup_audit

names = sys.argv[1:] or ["TMPR", "TMRD", "TRSAv1", "IMDB", "winvoker"]
allL = {**LOADERS, **LOADERS_EN}
results = {}
for n in names:
    t0 = time.time()
    d = core.add_keys(allL[n]())
    res, pairs, uniq = near_dup_audit(d)
    res["seconds"] = round(time.time() - t0, 1)
    results[n] = res
    # örnek çiftleri incelemek için kaydet
    ex = pairs.sort_values("jaccard").head(0)
    s = pairs.sample(min(30, len(pairs)), random_state=0) if len(pairs) else pairs
    pd.DataFrame({"a": uniq.k_norm.iloc[s.i].values, "b": uniq.k_norm.iloc[s.j].values,
                  "jaccard": s.jaccard.values,
                  "labels_a": uniq.labels.iloc[s.i].map(sorted).values,
                  "labels_b": uniq.labels.iloc[s.j].map(sorted).values}).to_csv(f"out/neardup_examples_{n}.csv", index=False)
    print(n, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    json.dump(results, open(f"out/neardup_{'_'.join(names)}.json", "w"), indent=1)
