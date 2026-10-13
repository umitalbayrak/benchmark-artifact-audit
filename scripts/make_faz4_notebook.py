"""Faz 4.3 Colab defteri: BERTurk ve XLM-R ince ayar; kirli / temiz / altın test tahminleri.
Kaldığı yerden devam eder: sonuçlar Google Drive'da 'faz4' klasörüne yazılır, tamamlanan koşular atlanır."""
import json

INTRO = """# Faz 4.3 — BERTurk ve XLM-R deneyleri

**Ne yapmanız gerekiyor:**

1. **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU → Kaydet.**
2. **Çalışma zamanı → Tümünü çalıştır.**
3. İkinci hücre Google Drive erişimi isteyecek: çıkan pencerede hesabınızı seçip **İzin ver** deyin. Sonuçlar Drive'ınızda `faz4` klasörüne kaydedilir; bağlantı koparsa hiçbir şey kaybolmaz.
4. Üçüncü hücre **Dosyaları seç** düğmesi gösterir: `faz4_data_1.zip` ve `faz4_data_2.zip` dosyalarının **ikisini birden** seçin. (Bu yalnızca ilk seferde gerekir; sonra Drive'dan okunur.)
5. Toplam süre GPU hızına göre **2–4 saat**. Sekme açık kalsın; bilgisayarın uyku moduna geçmemesi yeterli.
6. **Bağlantı koparsa veya Colab oturumu kapanırsa:** defteri yeniden açıp yine **Tümünü çalıştır** deyin. Tamamlanan koşular atlanır, kaldığı yerden devam eder. Dosya yükleme istenirse aynı iki zip'i tekrar seçin.
7. Bitince son hücre `faz4_results.zip` dosyasını indirir. Bu dosyayı deponun ilgili klasörüne açın (bkz. REPRODUCE.md).

Her koşu bittiğinde ekrana bir satır yazılır (ör. `berturk TRSAv1 s0 tamam — 6.2 dk`)."""

SETUP = """!pip -q install -U transformers accelerate
import torch, os; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK — T4 GPU seçin')"""

DRIVE = """from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/faz4'; DATA = OUT + '/data'
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT + '/runs', exist_ok=True)"""

UPLOAD = """import zipfile, glob
if len(glob.glob(DATA + '/*.parquet')) < 14:
    from google.colab import files
    up = files.upload()
    for name in up:
        zipfile.ZipFile(name).extractall(DATA)
print(sorted(os.path.basename(p) for p in glob.glob(DATA + '/*.parquet')))"""

RUN = r"""import json, time, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODELS = {'berturk': 'dbmdz/bert-base-turkish-cased', 'xlmr': 'FacebookAI/xlm-roberta-base'}
RUNS = [(d, s) for d in ['TMPR', 'TMRD'] for s in [0, 1, 2]] + [('TRSAv1', s) for s in [0, 1, 2]] + [('winvoker', s) for s in [0, 1, 2]]
WINVOKER_TRAIN_CAP = 100_000      # winvoker eğitimi tabakalı 100 bin örnekle sınırlı (süre); split sabit, tohum = model tohumu
MAXLEN, BS, LR = 128, 32, 2e-5
dev = 'cuda'

def batches(enc, labels, bs, shuffle, seed):
    idx = np.arange(len(labels));
    if shuffle: np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i+bs]
        yield {k: torch.tensor(v[j]).to(dev) for k, v in enc.items()}, torch.tensor(labels[j]).to(dev)

def encode(tok, texts):
    e = tok(list(map(str, texts)), truncation=True, max_length=MAXLEN, padding='max_length', return_tensors='np')
    return {k: e[k] for k in ('input_ids', 'attention_mask')}

@torch.no_grad()
def predict(model, tok, texts):
    model.eval(); out = []
    for i in range(0, len(texts), 256):
        e = tok(list(map(str, texts[i:i+256])), truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt').to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(model(**e).logits.argmax(-1).cpu().numpy())
    return np.concatenate(out)

for mname, mpath in MODELS.items():
    tok = AutoTokenizer.from_pretrained(mpath)
    for d, s in RUNS:
        tag = f'{mname}_{d}_s{s}'
        if os.path.exists(f'{OUT}/runs/{tag}.json'):
            print(tag, 'zaten tamam, atlanıyor'); continue
        t0 = time.time()
        split_seed = 0 if d == 'winvoker' else s
        x = pd.read_parquet(f'{DATA}/{d}_s{split_seed}.parquet'); gold = pd.read_parquet(f'{DATA}/gold_{d}.parquet')
        tr, te = x[x.part == 'train'], x[x.part == 'test'].reset_index(drop=True)
        if d == 'winvoker' and len(tr) > WINVOKER_TRAIN_CAP:
            tr = tr.groupby('label').sample(frac=WINVOKER_TRAIN_CAP / len(tr), random_state=s)
        labels = sorted(tr.label.unique()); l2i = {l: i for i, l in enumerate(labels)}
        y = tr.label.map(l2i).to_numpy()
        torch.manual_seed(s); np.random.seed(s)
        model = AutoModelForSequenceClassification.from_pretrained(mpath, num_labels=len(labels)).to(dev)
        epochs = 3 if len(tr) <= 20000 else 1
        enc = encode(tok, tr.text.tolist())
        steps = epochs * int(np.ceil(len(y) / BS))
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.cuda.amp.GradScaler()
        model.train()
        for ep in range(epochs):
            for xb, yb in batches(enc, y, BS, True, s * 100 + ep):
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = model(**xb, labels=yb).loss
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
                model.train()
        pt = predict(model, tok, te.text.tolist()); pg = predict(model, tok, gold.text.tolist())
        pd.DataFrame({'pred': [labels[i] for i in pt]}).to_parquet(f'{OUT}/runs/pred_{tag}.parquet', index=False)
        pd.DataFrame({'pred': [labels[i] for i in pg]}).to_parquet(f'{OUT}/runs/predgold_{tag}.parquet', index=False)
        acc = float(np.mean(np.array([labels[i] for i in pt]) == te.label.values))
        json.dump({'model': mname, 'dataset': d, 'seed': s, 'n_train': int(len(tr)), 'epochs': epochs,
                   'dirty_acc_quick': acc, 'minutes': round((time.time() - t0) / 60, 1)}, open(f'{OUT}/runs/{tag}.json', 'w'))
        print(f'{mname} {d} s{s} tamam — {round((time.time()-t0)/60,1)} dk, kirli doğruluk {acc:.3f}')
        del model, opt; torch.cuda.empty_cache()"""

DOWNLOAD = """import shutil
shutil.make_archive('/content/faz4_results', 'zip', OUT + '/runs')
from google.colab import files; files.download('/content/faz4_results.zip')"""

cells = [("markdown", INTRO), ("code", SETUP), ("code", DRIVE), ("code", UPLOAD), ("code", RUN), ("code", DOWNLOAD)]
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [dict(cell_type=k, metadata={}, source=v, **({"execution_count": None, "outputs": []} if k == "code" else {}))
                for k, v in cells]}
json.dump(nb, open("share/Faz4_Transformers.ipynb", "w"), ensure_ascii=False, indent=1)
for k, v in cells:
    if k == "code":
        compile(v.replace("!pip", "#pip"), "cell", "exec")
print("ok")
