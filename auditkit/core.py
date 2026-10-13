"""Denetim çekirdeği: normalizasyon, kopya, etiket çakışması, sızıntı, veri setleri arası örtüşme."""
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Türkçe karakterleri ASCII'ye katlama: TMRD gibi eski veri setleri ı/ş/ğ olmadan yazılmış,
# veri setleri arası örtüşmeyi yakalamak için "katlanmış" anahtar da üretilir.
_FOLD = str.maketrans("ıişğüöçâîû", "iisguocaiu")


def normalize(t: str) -> str:
    t = unicodedata.normalize("NFKC", str(t)).replace("İ", "i").replace("I", "ı").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def fold(t: str) -> str:
    return normalize(t).translate(_FOLD)


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["k_norm"] = df["text"].map(normalize)
    df["k_fold"] = df["text"].map(fold)
    df["n_words"] = df["k_norm"].str.split().str.len()
    return df


def duplicate_stats(df, key):
    """Fazlalık satır = toplam - benzersiz; boş anahtarlar (yalnız emoji/noktalama) ayrıca sayılır."""
    d = df[df[key] != ""]
    n = len(d)
    redundant = n - d[key].nunique()
    g = d.groupby(key)["label"].agg(["nunique", "size"])
    conflict = g[g["nunique"] > 1]
    return {
        "n_nonempty": n,
        "redundant_rows": redundant,
        "redundant_pct": 100 * redundant / n,
        "conflict_texts": len(conflict),
        "conflict_rows": int(conflict["size"].sum()),
        "conflict_rows_pct": 100 * conflict["size"].sum() / n,
    }


def split_leakage(df, key="k_norm"):
    """Resmî test satırlarından, metni eğitimde de geçenlerin oranı."""
    tr = set(df.loc[df.split == "train", key])
    te = df[(df.split == "test") & (df[key] != "")]
    hit = te[key].isin(tr)
    return {"test_rows": len(te), "leaked_rows": int(hit.sum()), "leaked_pct": 100 * hit.mean()}


def cv_leakage(df, key="k_norm", k=5, seed=0):
    """k katlı çapraz doğrulamada test katmanı satırlarının eğitim katmanında da geçme oranı."""
    d = df[df[key] != ""].reset_index(drop=True)
    rates = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(d, d["label"]):
        rates.append(d[key].iloc[te].isin(set(d[key].iloc[tr])).mean())
    return {"cv_leaked_pct": 100 * float(np.mean(rates))}


def cross_overlap(dfs: dict, key="k_fold", min_words=4):
    """Veri setleri arası örtüşme. Kısa kalıplar ("güzel ürün") doğal tekrar olduğundan
    yalnızca en az `min_words` kelimelik metinler sayılır."""
    keys = {n: set(d.loc[d.n_words >= min_words, key]) for n, d in dfs.items()}
    rows = []
    for a in dfs:
        for b in dfs:
            if a != b:
                inter = keys[a] & keys[b]
                rows.append({"dataset": a, "found_in": b, "shared_texts": len(inter),
                             "pct_of_dataset": 100 * len(inter) / max(len(keys[a]), 1)})
    return pd.DataFrame(rows)
