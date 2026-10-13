"""Veri seti yükleyicileri: her veri setini ortak şemaya çevirir.

Ortak şema: dataset, text, label (positive/negative/neutral), split (train/test/none), source
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
LABEL_MAP = {"positive": "positive", "negative": "negative", "neutral": "neutral", "notr": "neutral"}


def _decode(raw: bytes) -> str:
    """Satır bazında UTF-8, olmazsa cp1254 (Türkçe Windows) dener."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1254", errors="replace")


def _read_lines(path: Path):
    lines = [_decode(l).strip() for l in path.read_bytes().splitlines()]
    return [l for l in lines if l]


def load_winvoker():
    d = DATA / "c704dcc2-winvoker" / "winvoker"
    parts = []
    for split in ("train", "test"):
        df = pd.read_csv(d / f"{split}.csv")
        df = df.rename(columns={"dataset": "source"})
        df["split"] = split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["label"] = df["label"].str.lower().map(LABEL_MAP)
    df["dataset"] = "winvoker"
    return df[["dataset", "text", "label", "split", "source"]]


def load_trsav1():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "TRSAv1-Dataset" / "TRSAv1.csv")
    df = df.rename(columns={"review": "text", "score": "label"})
    df["label"] = df["label"].str.lower().map(LABEL_MAP)
    df["split"], df["source"], df["dataset"] = "none", "e-commerce", "TRSAv1"
    return df[["dataset", "text", "label", "split", "source"]]


def load_tmrd():
    d = DATA / "1c7419db-Turkish_Movie_Sentiment"
    rows = []
    for fn, lab in (("tr_polarity.pos", "positive"), ("tr_polarity.neg", "negative")):
        rows += [(t, lab) for t in _read_lines(d / fn)]
    df = pd.DataFrame(rows, columns=["text", "label"])
    df["split"], df["source"], df["dataset"] = "none", "movie", "TMRD"
    return df[["dataset", "text", "label", "split", "source"]]


def load_tmpr():
    d = DATA / "eeda0c8b-Turkish_Products_Sentiment"
    rows = []
    for dom in ("books", "dvd", "electronics", "kitchen"):
        for fn, lab in (("positive", "positive"), ("negative", "negative")):
            rows += [(t, lab, dom) for t in _read_lines(d / dom / fn)]
    df = pd.DataFrame(rows, columns=["text", "label", "source"])
    df["split"], df["dataset"] = "none", "TMPR"
    return df[["dataset", "text", "label", "split", "source"]]


def load_imdb():
    """IMDB (Maas vd. 2011) + Northcutt vd. (2021) MTurk doğrulaması.
    mturk_status: test örneğinde MTurk incelemesi varsa 'error' (5 işçiden çoğunluk verilen etiketi
    desteklemedi, yani given < 3; Northcutt vd. 2021'in IMDB için bildirdiği 725 hatayı birebir verir),
    'reviewed' (incelendi, hata doğrulanmadı) veya boş."""
    import json
    root = DATA / "imdb" / "aclImdb"
    rows = []
    for split in ("train", "test"):
        for lab in ("pos", "neg"):
            for f in sorted((root / split / lab).glob("*.txt")):
                rows.append((f.read_text(encoding="utf-8").replace("<br />", " "),
                             "positive" if lab == "pos" else "negative", split, f"{split}/{lab}/{f.stem}"))
    df = pd.DataFrame(rows, columns=["text", "label", "split", "doc_id"])
    mt = json.load(open(Path(__file__).resolve().parents[1] / "label-errors" / "mturk" / "imdb_mturk.json"))
    status = {m["id"]: ("error" if m["mturk"]["given"] < 3 else "reviewed") for m in mt}
    df["mturk_status"] = df["doc_id"].map(status)
    df["source"], df["dataset"] = "movie", "IMDB"
    return df[["dataset", "text", "label", "split", "source", "doc_id", "mturk_status"]]


LOADERS_EN = {"IMDB": load_imdb}

LOADERS = {"winvoker": load_winvoker, "TRSAv1": load_trsav1, "TMRD": load_tmrd, "TMPR": load_tmpr}
