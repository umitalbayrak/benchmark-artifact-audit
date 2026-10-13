"""Yakın kopya tespiti: MinHash LSH (karakter 5-gram shingle).

Kısa metinlerde (< min_words) yakın kopya anlamsız olduğundan yalnızca tam/normalize eşleşme kullanılır;
burada sadece >= min_words kelimelik benzersiz normalize metinler incelenir.
"""
import numpy as np
import pandas as pd
from datasketch import LeanMinHash, MinHash, MinHashLSH

NUM_PERM = 64
SEED = 1


def _shingles(text, k=5):
    """Karakter 5-gram: kısa yorumlarda ve Türkçe eklemeli yapıda kelime n-gram'ından daha kararlı."""
    return {text[i:i + k].encode("utf-8") for i in range(max(len(text) - k + 1, 1))}


def near_duplicate_pairs(texts, threshold=0.7, k=5):
    """texts: benzersiz normalize metin listesi. Döner: (i, j, tahmini_jaccard) çiftleri."""
    # Bellek: shingle kümeleri ve MinHash nesneleri tutulmaz; yalnızca LSH tabloları tutulur.
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    pairs = []
    batch = 20000
    for s in range(0, len(texts), batch):
        mhs = MinHash.bulk((_shingles(t, k) for t in texts[s:s + batch]), num_perm=NUM_PERM, seed=SEED)
        for off, m in enumerate(mhs):
            i = s + off
            for j in lsh.query(m):
                pairs.append((j, i))
            lsh.insert(i, LeanMinHash(m))
        del mhs
    # LSH aday üretir; gerçek Jaccard ile doğrula (shingle kümeleri yalnızca adaylar için yeniden üretilir)
    out = []
    for i, j in pairs:
        a, b = _shingles(texts[i], k), _shingles(texts[j], k)
        jac = len(a & b) / len(a | b) if (a or b) else 0.0
        if jac >= threshold:
            out.append((i, j, jac))
    return pd.DataFrame(out, columns=["i", "j", "jaccard"])


def near_dup_audit(df, min_words=5, thresholds=(0.7, 0.8, 0.9)):
    """df: add_keys uygulanmış veri. Normalize metin başına bir temsilci üzerinden çalışır."""
    d = df[df.n_words >= min_words]
    uniq = d.groupby("k_norm").agg(labels=("label", lambda s: frozenset(s)),
                                   splits=("split", lambda s: frozenset(s)),
                                   rows=("label", "size")).reset_index()
    texts = uniq["k_norm"].tolist()
    pairs = near_duplicate_pairs(texts, threshold=min(thresholds))
    res = {"unique_texts": len(uniq), "pairs_found": len(pairs)}
    for t in thresholds:
        p = pairs[pairs.jaccard >= t]
        involved = set(p.i) | set(p.j)
        lab_conf = sum(uniq.labels.iat[i] != uniq.labels.iat[j] for i, j in zip(p.i, p.j))
        res[f"pct_texts_with_neardup@{t}"] = 100 * len(involved) / max(len(uniq), 1)
        res[f"label_conflict_pairs@{t}"] = int(lab_conf)
        # train-test yakın kopya sızıntısı: test metni, eğitimdeki farklı bir metne yakın
        if (df.split == "test").any():
            te_only = uniq.splits.map(lambda s: s == frozenset({"test"}))
            tr_any = uniq.splits.map(lambda s: "train" in s)
            leak = set()
            for i, j in zip(p.i, p.j):
                if te_only.iat[i] and tr_any.iat[j]:
                    leak.add(i)
                if te_only.iat[j] and tr_any.iat[i]:
                    leak.add(j)
            n_test = int(te_only.sum())
            res[f"test_texts_neardup_in_train@{t}"] = len(leak)
            res[f"test_neardup_leak_pct@{t}"] = 100 * len(leak) / max(n_test, 1)
    return res, pairs, uniq
