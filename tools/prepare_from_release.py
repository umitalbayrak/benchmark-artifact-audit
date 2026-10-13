"""Paylaşılan indeks/özet dosyalarından, kullanıcının kendi indirdiği ham verilerle çalışma dosyalarını yeniden kurar.

Önce data/README.md'ye göre ham verileri yerleştirin; sonra depo kökünden:
    PYTHONPATH=. python tools/prepare_from_release.py
Üretilenler (orijinal analizdeki adlarla):
    out/annotation_final.csv, out/annotation_population.json   (insan doğrulama; metinler yerel veriden)
    out/embed_meta.parquet, out/embed_pairs.csv.gz                (gömme yakın kopya çiftleri)
    out/splits/<d>_s<seed>.parquet, out/splits/gold_<d>.parquet   (kirli/temiz/altın test tanımları)
    faz4_in/, faz4_llm/, out/splits/pred*_tfidf*, faz5_in/, out/imdb_preds.parquet (model tahminleri)
Her metin, paylaşılan SHA-1 özetiyle doğrulanır; tek bir uyuşmazlıkta durur."""
import hashlib, json, shutil, sys
from pathlib import Path
import pandas as pd
from auditkit.loaders import LOADERS, LOADERS_EN
from auditkit import core

REL = Path("release_data")
OUT = Path("out"); (OUT / "splits").mkdir(parents=True, exist_ok=True)
def h16(s): return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:16]

loaded = {}
def get(d):
    if d not in loaded:
        x = core.add_keys({**LOADERS, **LOADERS_EN}[d]())
        x["kh"] = x.k_norm.map(h16); x["th"] = x.text.map(h16)
        loaded[d] = x
    return loaded[d]

# 1) İnsan doğrulama dosyası (metin, özgün etiket ve kaynak yerel kopyadan, orig_idx ile)
ann = pd.read_csv(REL / "annotation/annotations.csv")
rows = []
for d, a in ann.groupby("dataset", sort=False):
    x = get(d)
    for _, r in a.iterrows():
        m = x.iloc[int(r.orig_idx)]
        if m.th != r.text_sha1_16 or m.kh != r.k_norm_sha1_16:
            sys.exit(f"Metin özeti uyuşmazlığı: {r.item_id} ({d}). Ham veri sürümünüz farklı olabilir; data/README.md'ye bakın.")
        rows.append({"dataset": d, "stratum": r.stratum, "text": m.text, "k_norm": m.k_norm, "orig_label": m.label,
                     "source": m.source, "pair_id": r.pair_id, "show_id": r.show_order, "item_id": r.item_id,
                     "A": r.annotator_1, "B": r.annotator_2, "agree": r.agree,
                     "gold": r.annotator_1 if r.agree else None, "verdict": r.adjudication_verdict,
                     "adj": r.adjudicated_label, "gold_final": r.gold_final, "source_of_gold": r.source_of_gold})
fin = pd.DataFrame(rows); fin.to_csv(OUT / "annotation_final.csv", index=False)
shutil.copy(REL / "annotation/annotation_population.json", OUT / "annotation_population.json")
print("annotation_final.csv:", len(fin), "madde", flush=True)

# 2) Gömme çiftleri (meta, build_embed_bundle.py ile aynı kural: >= 5 kelime, benzersiz normalize metin, sıralı)
mh = pd.read_csv(REL / "embeddings/embed_meta_hashes.csv.gz")
meta = []
for d in mh.dataset.unique():
    x = get(d); x = x[x.n_words >= 5]
    u = x.groupby("k_norm").agg(text=("text", "first"), labels=("label", lambda s: "|".join(sorted(set(s)))),
                                splits=("split", lambda s: "|".join(sorted(set(s))))).reset_index()
    u.insert(0, "id", range(len(u))); u.insert(0, "dataset", d)
    ref = mh[mh.dataset == d]
    if len(ref) != len(u) or not (u.k_norm.map(h16).values == ref.k_norm_sha1_16.values).all():
        sys.exit(f"Gömme meta uyuşmazlığı: {d}")
    meta.append(u)
pd.concat(meta).to_parquet(OUT / "embed_meta.parquet", index=False)
shutil.copy(REL / "embeddings/embed_pairs.csv.gz", OUT / "embed_pairs.csv.gz")
print("embed_meta ok", flush=True)

# 3) Bölümler ve altın kümeler
for f in sorted((REL / "splits").glob("*_s*.csv.gz")):
    d = f.name.split("_s")[0]; sp = pd.read_csv(f, keep_default_na=False)
    x = get(d).iloc[sp.orig_idx.values]
    if not (x.th.values == sp.text_sha1_16.values).all():
        sys.exit(f"Bölüm metin özeti uyuşmazlığı: {f.name}")
    pd.DataFrame({"text": x.text.values, "label": x.label.values, "part": sp.part.values, "k_norm": x.k_norm.values,
                  "source": x.source.values, "clean_drop_reason": sp.clean_drop_reason.values}) \
        .to_parquet(OUT / "splits" / f.name.replace(".csv.gz", ".parquet"), index=False)
for f in sorted((REL / "splits").glob("gold_*.csv")):
    g = pd.read_csv(f).merge(fin[["item_id", "text", "orig_label"]].drop_duplicates("item_id"), on="item_id", how="left")
    if not (g.text.map(h16) == g.text_sha1_16).all():
        sys.exit(f"Altın küme uyuşmazlığı: {f.name}")
    g = g.rename(columns={"gold_label": "label"})
    g[["text", "label", "orig_label", "stratum", "weight"]].to_parquet(OUT / "splits" / f.name.replace(".csv", ".parquet"), index=False)
shutil.copy(REL / "splits/summary.json", OUT / "splits/summary.json")
print("splits ok", flush=True)

# 4) Model tahminleri
P = REL / "predictions"
for src, dst in [("faz4_transformers", "faz4_in"), ("faz4_llm", "faz4_llm"), ("tfidf", "out/splits")]:
    Path(dst).mkdir(parents=True, exist_ok=True)
    for f in (P / src).iterdir(): shutil.copy(f, Path(dst) / f.name)
# IMDB: özgün etiket yerel kopyadan (doc_id ile), Northcutt düzeltilmiş etiket corr_status'tan
Path("faz5_in").mkdir(exist_ok=True); Path("share").mkdir(exist_ok=True)
imdb = get("IMDB").query("split == 'test'").set_index("doc_id")
flip = {"positive": "negative", "negative": "positive"}
def add_labels(t, key):
    lab = (t[key].map(imdb.label) if key == "doc_id" else t[key].map(imdb.reset_index().assign(h=lambda z: z.text.map(
        lambda u: hashlib.md5(u.encode("utf-8")).hexdigest())).drop_duplicates("h").set_index("h").label))
    if lab.isna().any(): sys.exit("IMDB eşleşme hatası")
    t.insert(2, "label", lab.values)
    t.insert(3, "label_corr", [flip[l] if c == "flipped" else (None if c == "dropped_ambiguous" else l)
                               for l, c in zip(t.label, t.corr_status)])
    return t
for f in (P / "imdb").iterdir():
    if f.name == "imdb_preds_classical.parquet":
        add_labels(pd.read_parquet(f), "doc_id").to_parquet(OUT / "imdb_preds.parquet", index=False)
    elif f.name == "imdb_test_keys.parquet":
        k = add_labels(pd.read_parquet(f), "h")
        k[["h", "label", "label_corr", "corr_status", "clean_drop_reason", "hn"]].to_parquet("share/imdb_colab_keys.parquet", index=False)
    else: shutil.copy(f, Path("faz5_in") / f.name)
for f in ["model_results.csv", "tfidf_results.json", "cleanlab_TMPR_TMRD_IMDB_TRSAv1.json", "neardup_IMDB_winvoker.json", "embed_neardup.json"]:
    shutil.copy(REL / "results" / f, OUT / f)
print("Hazır. Örnek: PYTHONPATH=. python scripts/run_eval_models.py ; PYTHONPATH=. python tools/verify_release.py")
