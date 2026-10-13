"""2.2 gömme bileşeni için Colab paketi: her veri setinde >=5 kelimelik benzersiz normalize metinler
(MinHash ile aynı küme), her biri için ilk görülen ham metin temsilcisi. Eşleme dosyası yerelde kalır."""
import pandas as pd
from auditkit.loaders import LOADERS, LOADERS_EN
from auditkit import core
rows, meta = [], []
for n, f in {**LOADERS, **LOADERS_EN}.items():
    d = core.add_keys(f()); d = d[d.n_words >= 5]
    u = d.groupby("k_norm").agg(text=("text", "first"), labels=("label", lambda s: "|".join(sorted(set(s)))),
                                splits=("split", lambda s: "|".join(sorted(set(s))))).reset_index()
    u.insert(0, "id", range(len(u))); u.insert(0, "dataset", n)
    rows.append(u[["dataset", "id", "text"]]); meta.append(u)
    print(n, len(u))
pd.concat(rows).to_parquet("share/embed_input.parquet", index=False, compression="zstd")
pd.concat(meta).to_parquet("out/embed_meta.parquet", index=False)
