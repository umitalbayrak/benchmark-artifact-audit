"""Faz 2.2 gömme bileşeni için Colab defterini üretir: share/Faz2_2_Embedding_NearDup.ipynb"""
import json

cells = [
    ("markdown", """# Faz 2.2 — Gömme tabanlı yakın kopya analizi

**Ne yapmanız gerekiyor (yaklaşık 15 dakika):**

1. Üst menüden **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU** seçin ve kaydedin.
2. **Çalışma zamanı → Tümünü çalıştır** deyin.
3. Üçüncü hücre bir **Dosya seç** düğmesi gösterecek: `embed_input_1.parquet` ve `embed_input_2.parquet` dosyalarının **ikisini birden** seçin (yüklenmesi birkaç dakika sürebilir).
4. Bitince `embed_pairs.csv.gz` dosyası otomatik inecek. Bu dosyayı deponun ilgili klasörüne açın (bkz. REPRODUCE.md).

Başka hiçbir ayar gerekmiyor. Model: `paraphrase-multilingual-MiniLM-L12-v2` (Türkçe dahil 50+ dil)."""),
    ("code", """!pip -q install sentence-transformers
import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"""),
    ("code", """from google.colab import files
import pandas as pd
up = files.upload()
df = pd.concat([pd.read_parquet(f) for f in ['embed_input_1.parquet', 'embed_input_2.parquet']], ignore_index=True)
print(df.dataset.value_counts())"""),
    ("code", """from sentence_transformers import SentenceTransformer
import torch, numpy as np, gzip, csv, time
MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
TOPK, MIN_COS = 10, 0.90
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(MODEL, device=dev); model.max_seq_length = 256

out = gzip.open('embed_pairs.csv.gz', 'wt', newline='')
w = csv.writer(out); w.writerow(['dataset', 'i', 'j', 'cos'])
for name, g in df.groupby('dataset', sort=False):
    t0 = time.time()
    ids = g['id'].to_numpy(); texts = g['text'].tolist()
    emb = model.encode(texts, batch_size=256, convert_to_tensor=True, normalize_embeddings=True,
                       show_progress_bar=True)
    emb = emb.half() if dev == 'cuda' else emb
    n, kept = emb.shape[0], 0
    for s in range(0, n, 4096):
        sims = emb[s:s+4096] @ emb.T
        rows = torch.arange(s, min(s+4096, n), device=emb.device)
        sims[torch.arange(len(rows)), rows] = -1            # kendisiyle eşleşmeyi çıkar
        val, idx = sims.topk(min(TOPK, n-1), dim=1)
        val, idx = val.float().cpu().numpy(), idx.cpu().numpy()
        for r in range(len(rows)):
            a = s + r
            for v, b in zip(val[r], idx[r]):
                if v < MIN_COS: break
                lo, hi = (a, b) if a < b else (b, a)
                w.writerow([name, int(ids[lo]), int(ids[hi]), round(float(v), 4)]); kept += 1
    print(name, n, 'metin,', kept, 'çift (tekrarlar yerelde temizlenecek),', round(time.time()-t0), 'sn')
    del emb; torch.cuda.empty_cache()
out.close()"""),
    ("code", """files.download('embed_pairs.csv.gz')"""),
]

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": []}
for kind, src in cells:
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    nb["cells"].append(c)
json.dump(nb, open("share/Faz2_2_Embedding_NearDup.ipynb", "w"), ensure_ascii=False, indent=1)
print("ok")
