"""Faz 5.1: aynı denetim çerçevesinin İngilizce IMDB'de çalıştırılması ve Northcutt vd. (2021) düzeltmeleriyle karşılaştırma.

Amaç yalnız dış geçerlilik: Türkçe bulgulardaki üç mekanizma anlatısı burada zorlanmaz; çerçeve IMDB'de ne gösteriyorsa o raporlanır.

Test türleri (resmî train/test bölümü, 25k/25k):
  dirty      resmî test, olduğu gibi
  clean      Türkçe protokolle aynı: test eksi (a) eğitimde birebir/normalize geçen, (b) eğitimdeki bir metinle
             yakın/anlamsal kopyası olan (MinHash J >= 0,8 veya gömme cos >= 0,98), (c) etiket çakışan, (d) boş
  corrected  Northcutt düzeltmeli test: MTurk'te hata doğrulanan 725 örnekten (given < 3) çoğunluğun karşı etiketi
             seçtiği 173'ü (guessed >= 3) çevrilir, kalan 552'si (belirsiz/nötr/konu dışı) çıkarılır
  clean_corrected  clean ∩ corrected
Ek: MTurk'te incelenen 1.310 örnekte (Northcutt'un aday kümesi; bizim A tabakasının karşılığı) özgün ve düzeltilmiş
etikete göre doğruluk.
Modeller (yerel, CPU): TF-IDF+LR (ana model, Türkçe deneylerle aynı), TF-IDF+LinearSVC (kelime), MultinomialNB (kelime 1-2).
İstatistik: tabakasız bootstrap (2.000) GA; aynı tahminlerle dirty vs corrected fark GA; model çiftleri McNemar.
Çıktılar: out/imdb_results.json, out/imdb_preds.parquet, share/imdb_colab_keys.parquet (Colab transformer defteri için)."""
import hashlib, json, itertools, time
import numpy as np, pandas as pd
from scipy.stats import binomtest, kendalltau
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from auditkit.loaders import load_imdb
from auditkit import core
from auditkit.models import tfidf_lr
from auditkit.neardup import near_duplicate_pairs

t0 = time.time()
mt = {m["id"]: m["mturk"] for m in json.load(open("label-errors/mturk/imdb_mturk.json"))}
df = core.add_keys(load_imdb()).reset_index(drop=True)
df["h"] = df.text.map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
out = {}

# ---------- Denetim özeti (Faz 2 çerçevesi)
dup = core.duplicate_stats(df, "k_norm"); leak = core.split_leakage(df, "k_norm")
conflict = set(df.groupby("k_norm").label.nunique().loc[lambda s: s > 1].index)
out["audit"] = {"rows": len(df), "dup": dup, "split_leakage": leak, "label_conflict_texts": len(conflict),
                "neardup_minhash": json.load(open("out/neardup_IMDB_winvoker.json"))["IMDB"],
                "neardup_embed": json.load(open("out/embed_neardup.json"))["IMDB"],
                "cleanlab_vs_northcutt": json.load(open("out/cleanlab_TMPR_TMRD_IMDB_TRSAv1.json"))["IMDB"]}

# ---------- Temiz test
emb = pd.read_csv("out/embed_pairs.csv.gz").query("dataset == 'IMDB' and cos >= 0.98").drop_duplicates(["i", "j"])
meta = pd.read_parquet("out/embed_meta.parquet").query("dataset == 'IMDB'").set_index("id").k_norm
nbr = {}
for i, j in zip(emb.i, emb.j):
    a, b = meta.loc[i], meta.loc[j]; nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)
u = sorted(set(df.loc[df.n_words >= 5, "k_norm"]))
for _, r in near_duplicate_pairs(u, threshold=0.8).iterrows():
    a, b = u[int(r.i)], u[int(r.j)]; nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)
train_keys = set(df.loc[df.split == "train", "k_norm"])
te = df[df.split == "test"].copy()
reason = pd.Series("", index=te.index)
reason[te.k_norm.eq("")] = "empty"
reason[reason.eq("") & te.k_norm.isin(conflict)] = "label_conflict"
reason[reason.eq("") & te.k_norm.isin(train_keys)] = "exact_in_train"
reason[reason.eq("") & te.k_norm.map(lambda k: bool(nbr.get(k, set()) & train_keys))] = "neardup_in_train"
te["clean_drop_reason"] = reason

# ---------- Northcutt düzeltmesi
def corr(row):
    m = mt.get(row.doc_id)
    if m is None or m["given"] >= 3:
        return row.label, "kept" if m is None else "reviewed_ok"
    if m["guessed"] >= 3:
        return ("negative" if row.label == "positive" else "positive"), "flipped"
    return None, "dropped_ambiguous"
cc = te.apply(corr, axis=1, result_type="expand"); te["label_corr"], te["corr_status"] = cc[0], cc[1]
te["reviewed"] = te.doc_id.isin(mt)
out["tests"] = {"dirty": len(te), "clean": int(te.clean_drop_reason.eq("").sum()),
                "clean_dropped": te.clean_drop_reason[te.clean_drop_reason != ""].value_counts().to_dict(),
                "corr_status": te.corr_status.value_counts().to_dict(), "mturk_reviewed": int(te.reviewed.sum())}
print(json.dumps(out["tests"]), flush=True)

# ---------- Modeller
tr = df[df.split == "train"]
MODELS = {
    "tfidf_lr": lambda: tfidf_lr(),
    "tfidf_svm": lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LinearSVC(C=0.5)),
    "nb": lambda: make_pipeline(CountVectorizer(ngram_range=(1, 2), min_df=2), MultinomialNB(alpha=0.5)),
}
for name, mk in MODELS.items():
    te[f"pred_{name}"] = mk().fit(tr.text, tr.label).predict(te.text)
    print(name, "fit", round(time.time() - t0), "s", flush=True)

def subsets(t):
    cl = t.clean_drop_reason.eq(""); co = t.label_corr.notna()
    return {"dirty": (t.index, "label"), "clean": (t.index[cl], "label"),
            "corrected": (t.index[co], "label_corr"), "clean_corrected": (t.index[cl & co], "label_corr")}

def mf1(y, p): return f1_score(y, p, average="macro")
S = subsets(te); M = list(MODELS)
out["scores"] = {m: {k: {"acc": float((te.loc[ix, f"pred_{m}"] == te.loc[ix, lab]).mean()),
                         "macro_f1": float(mf1(te.loc[ix, lab], te.loc[ix, f"pred_{m}"]))}
                     for k, (ix, lab) in S.items()} for m in M}

# Bootstrap: ortak örneklem (test satırları), her örnekte tüm test türleri yeniden hesaplanır
rng = np.random.default_rng(0); B = 2000
lab = te.label.values; labc = te.label_corr.values; cl = te.clean_drop_reason.eq("").values; co = te.label_corr.notna().values
P = {m: te[f"pred_{m}"].values for m in M}
boot = {m: {k: [] for k in S} for m in M}
for _ in range(B):
    ii = rng.integers(0, len(te), len(te))
    for m in M:
        p = P[m][ii]
        boot[m]["dirty"].append(np.mean(p == lab[ii]))
        boot[m]["clean"].append(np.mean((p == lab[ii])[cl[ii]]))
        boot[m]["corrected"].append(np.mean((p == labc[ii])[co[ii]]))
        boot[m]["clean_corrected"].append(np.mean((p == labc[ii])[co[ii] & cl[ii]]))
boot = {m: {k: np.array(v) for k, v in d.items()} for m, d in boot.items()}
ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
out["acc_ci"] = {m: {k: ci(v) for k, v in d.items()} for m, d in boot.items()}
out["corrected_minus_dirty_acc"] = {m: [round(out["scores"][m]["corrected"]["acc"] - out["scores"][m]["dirty"]["acc"], 4),
                                        *ci(boot[m]["corrected"] - boot[m]["dirty"])] for m in M}
out["clean_minus_dirty_acc"] = {m: [round(out["scores"][m]["clean"]["acc"] - out["scores"][m]["dirty"]["acc"], 4),
                                    *ci(boot[m]["clean"] - boot[m]["dirty"])] for m in M}

# Sıralama ve çiftler
rank = {k: sorted(M, key=lambda m: -out["scores"][m][k]["acc"]) for k in S}
dirty_v = [out["scores"][m]["dirty"]["acc"] for m in M]
out["ranking"] = rank
out["kendall"] = {k: float(kendalltau(dirty_v, [out["scores"][m][k]["acc"] for m in M]).statistic) for k in S}
od = list(np.argsort(-np.array(dirty_v)))
out["boot_share_rank_equal_dirty"] = {k: float(np.mean([list(np.argsort(-np.array([boot[m][k][b] for m in M]))) == od
                                                        for b in range(B)])) for k in S}
out["mcnemar"] = {}
for a, b in itertools.combinations(M, 2):
    r = {}
    for k, (ix, lb) in S.items():
        y = te.loc[ix, lb].values; ca = te.loc[ix, f"pred_{a}"].values == y; cb = te.loc[ix, f"pred_{b}"].values == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        r[k] = {"a_only": n01, "b_only": n10, "p": float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0}
    out["mcnemar"][f"{a}-{b}"] = r

# Northcutt aday kümesi (1.310 incelenen): özgün etikete göre vs düzeltilmiş etikete göre
rv = te[te.reviewed]
out["reviewed_subset"] = {}
for m in M:
    p = rv[f"pred_{m}"]; ok = rv.label_corr.notna()
    out["reviewed_subset"][m] = {
        "acc_vs_given": float((p == rv.label).mean()),
        "acc_vs_corrected": float((p[ok] == rv.label_corr[ok]).mean()),
        "flipped_n": int((rv.corr_status == "flipped").sum()),
        "flipped_pred_agrees_corrected": float((p == rv.label_corr)[rv.corr_status == "flipped"].mean()),
        "ambiguous_n": int((rv.corr_status == "dropped_ambiguous").sum()),
        "ambiguous_pred_agrees_given": float((p == rv.label)[rv.corr_status == "dropped_ambiguous"].mean())}

# Sızıntı alt kümesinde doğruluk (yön kontrolü)
leak_ix = te.index[te.clean_drop_reason.isin(["exact_in_train", "neardup_in_train"])]
out["leaked_subset_acc"] = {m: {"n": len(leak_ix), "acc": float((te.loc[leak_ix, f"pred_{m}"] == te.loc[leak_ix, "label"]).mean())
                                if len(leak_ix) else None} for m in M}
out["seconds"] = round(time.time() - t0)

te[["doc_id", "h", "label", "label_corr", "corr_status", "clean_drop_reason"] + [f"pred_{m}" for m in M]] \
    .to_parquet("out/imdb_preds.parquet", index=False)
te[["h", "label", "label_corr", "corr_status", "clean_drop_reason"]].to_parquet("share/imdb_colab_keys.parquet", index=False)
json.dump(out, open("out/imdb_results.json", "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ["scores", "corrected_minus_dirty_acc", "clean_minus_dirty_acc", "ranking", "kendall",
                                      "boot_share_rank_equal_dirty", "reviewed_subset", "leaked_subset_acc"]}, indent=1, default=float))
