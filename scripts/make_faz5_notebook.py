"""Faz 5.1 Colab defteri: IMDB'de İngilizce transformer (BERT-base, RoBERTa-base) ince ayarı + isteğe bağlı sıfır-atış LLM.
Veri HF'den (stanfordnlp/imdb) indirilir; yerel test satırlarıyla md5 metin özetiyle eşleştirilir.
Kaldığı yerden devam eder (Drive'da 'faz5' klasörü)."""
import json

INTRO = """# Faz 5.1 — IMDB: İngilizce transformer deneyleri

**Ne yapmanız gerekiyor:**

1. **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU → Kaydet.**
2. **Çalışma zamanı → Tümünü çalıştır.**
3. Drive izni isterse **İzin ver** deyin (sonuçlar Drive'da `faz5` klasörüne yazılır; bağlantı koparsa kaybolmaz).
4. **Dosyaları seç** düğmesi çıkınca yalnızca `imdb_colab_keys.parquet` dosyasını seçin (yalnızca ilk seferde).
5. Süre yaklaşık **1,5–2,5 saat** (2 model × 3 tohum + LLM). Önce tüm modellerin tohum 0 koşuları yapılır.
   Süre yetmezse ya da GPU kotası biterse: **son hücreyi (indirme) elle çalıştırın**, o ana kadar biten koşular iner; yeterlidir.
6. Bağlantı koparsa defteri yeniden açıp **Tümünü çalıştır** deyin; biten koşular atlanır.
7. Bitince `faz5_results.zip` iner; deponun ilgili klasörüne açın (bkz. REPRODUCE.md)."""

SETUP = """!pip -q install -U transformers accelerate datasets bitsandbytes
import torch, os; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK — T4 GPU seçin')
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/faz5'; RUNS = OUT + '/runs'
os.makedirs(RUNS, exist_ok=True)"""

DATA = r"""import hashlib, numpy as np, pandas as pd
from datasets import load_dataset
if not os.path.exists(OUT + '/imdb_colab_keys.parquet'):
    from google.colab import files
    up = files.upload()
    for name in up:
        import shutil; shutil.copy(name, OUT + '/imdb_colab_keys.parquet')
KEYS = pd.read_parquet(OUT + '/imdb_colab_keys.parquet')
ds = load_dataset('stanfordnlp/imdb')
def frame(split):
    d = ds[split].to_pandas()
    d['text'] = d.text.str.replace('<br />', ' ', regex=False)
    d['h'] = d.text.map(lambda t: hashlib.md5(t.encode('utf-8')).hexdigest())
    d['hn'] = d.text.map(lambda t: hashlib.md5(' '.join(t.split()).encode('utf-8')).hexdigest())
    d['label'] = np.where(d.label == 1, 'positive', 'negative')
    return d[['text', 'label', 'h', 'hn']]
TR, TE = frame('train'), frame('test')
KEY = 'h' if TE.h.isin(set(KEYS.h)).mean() > 0.99 else 'hn'      # boşluk farkı varsa yedek özet
keys = KEYS.drop_duplicates(KEY).set_index(KEY)
match = TE[KEY].isin(keys.index).mean(); print('Eşleşme anahtarı:', KEY)
print('Test eşleşme oranı:', round(match, 4), '| eğitim', len(TR), '| test', len(TE))
assert match > 0.99, 'Eşleşme düşük — veri sürümünü kontrol edin'"""

RUN = r"""import json, time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
MODELS = {'bert': 'google-bert/bert-base-uncased', 'roberta': 'FacebookAI/roberta-base'}
ORDER = [(m, s) for s in [0, 1, 2] for m in MODELS]        # önce tüm modellerin tohum 0'ı
MAXLEN, BS, LR, EPOCHS = 256, 16, 2e-5, 1                   # Türkçe protokol: > 20 bin eğitim örneğinde 1 epok
dev = 'cuda'; LABELS = ['negative', 'positive']; l2i = {l: i for i, l in enumerate(LABELS)}

@torch.no_grad()
def predict(model, tok, texts):
    model.eval(); out = []
    for i in range(0, len(texts), 128):
        e = tok(texts[i:i+128], truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt').to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(model(**e).logits.argmax(-1).cpu().numpy())
    return np.concatenate(out)

for mname, s in ORDER:
    tag = f'{mname}_s{s}'
    if os.path.exists(f'{RUNS}/{tag}.json'):
        print(tag, 'zaten tamam'); continue
    t0 = time.time(); tok = AutoTokenizer.from_pretrained(MODELS[mname])
    torch.manual_seed(s); np.random.seed(s)
    model = AutoModelForSequenceClassification.from_pretrained(MODELS[mname], num_labels=2).to(dev)
    enc = tok(TR.text.tolist(), truncation=True, max_length=MAXLEN, padding='max_length', return_tensors='np')
    y = TR.label.map(l2i).to_numpy(); steps = EPOCHS * int(np.ceil(len(y) / BS))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps); scaler = torch.cuda.amp.GradScaler()
    model.train()
    for ep in range(EPOCHS):
        idx = np.random.default_rng(s * 100 + ep).permutation(len(y))
        for i in range(0, len(idx), BS):
            j = idx[i:i+BS]
            xb = {k: torch.tensor(enc[k][j]).to(dev) for k in ('input_ids', 'attention_mask')}
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(**xb, labels=torch.tensor(y[j]).to(dev)).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
    p = predict(model, tok, TE.text.tolist())
    pd.DataFrame({'h': TE.h, 'hn': TE.hn, 'pred': [LABELS[i] for i in p]}).to_parquet(f'{RUNS}/pred_{tag}.parquet', index=False)
    acc = float(np.mean(np.array([LABELS[i] for i in p]) == TE.label.values))
    json.dump({'model': mname, 'seed': s, 'maxlen': MAXLEN, 'epochs': EPOCHS, 'dirty_acc_quick': acc,
               'minutes': round((time.time() - t0) / 60, 1)}, open(f'{RUNS}/{tag}.json', 'w'))
    print(f'{tag} tamam — {round((time.time()-t0)/60,1)} dk, kirli doğruluk {acc:.3f}')
    del model, opt; torch.cuda.empty_cache()"""

LLM = r"""# İkincil: sıfır-atış LLM (MTurk'te incelenen tüm test örnekleri + 2.000 rastgele test örneği)
import json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
if not os.path.exists(f'{RUNS}/llm.parquet'):
    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
    for MODEL in ['Qwen/Qwen3-8B', 'Qwen/Qwen2.5-7B-Instruct']:
        try:
            tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = 'left'
            model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map='auto'); model.eval(); break
        except Exception as e:
            print('Yüklenemedi:', MODEL, repr(e)[:200])
    k = keys.reindex(TE[KEY])
    rev = TE[k.corr_status.isin(['flipped', 'dropped_ambiguous', 'reviewed_ok']).values]
    rnd = TE[~TE[KEY].isin(rev[KEY])].sample(n=2000, random_state=0)
    sub = pd.concat([rev.assign(part='reviewed'), rnd.assign(part='random')]).drop_duplicates(KEY)
    ids = torch.tensor([tok.encode(c, add_special_tokens=False)[0] for c in 'AB']).to(model.device)
    def prompt(t):
        msg = (f'Determine the overall sentiment of the following movie review.\n\nReview: {t[:2500]}\n\n'
               'Options:\nA) Positive\nB) Negative\n\nAnswer with the option letter only.')
        kw = {'enable_thinking': False} if 'Qwen3' in MODEL else {}
        return tok.apply_chat_template([{'role': 'user', 'content': msg}], tokenize=False, add_generation_prompt=True, **kw)
    preds = []
    with torch.no_grad():
        for i in range(0, len(sub), 8):
            enc = tok([prompt(t) for t in sub.text.iloc[i:i+8]], return_tensors='pt', padding=True).to(model.device)
            preds += ['positive' if j == 0 else 'negative' for j in model(**enc).logits[:, -1, :][:, ids].argmax(-1).tolist()]
    sub.assign(pred=preds, model=MODEL)[['h', 'hn', 'part', 'pred', 'model']].to_parquet(f'{RUNS}/llm.parquet', index=False)
    print('LLM tamam —', len(sub), 'örnek,', round((time.time() - t0) / 60, 1), 'dk')"""

DOWNLOAD = """import shutil
shutil.make_archive('/content/faz5_results', 'zip', RUNS)
from google.colab import files; files.download('/content/faz5_results.zip')"""

cells = [("markdown", INTRO), ("code", SETUP), ("code", DATA), ("code", RUN), ("code", LLM), ("code", DOWNLOAD)]
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [dict(cell_type=k, metadata={}, source=v, **({"execution_count": None, "outputs": []} if k == "code" else {}))
                for k, v in cells]}
json.dump(nb, open("share/Faz5_IMDB.ipynb", "w"), ensure_ascii=False, indent=1)
for k, v in cells:
    if k == "code":
        compile(v.replace("!pip", "#pip"), "cell", "exec")
print("ok")
