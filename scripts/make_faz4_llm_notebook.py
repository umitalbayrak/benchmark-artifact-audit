"""Faz 4.4 Colab defteri: açık LLM sıfır-atış (ikincil karşılaştırma). Faz 4.3'ün Drive'daki verisini kullanır.
Skorlama: seçenek harfi (A/B/C) için sonraki-token logit karşılaştırması; üretim yok, deterministik."""
import json

INTRO = """# Faz 4.4 — Açık LLM sıfır-atış karşılaştırması

**Ne yapmanız gerekiyor (yaklaşık 30–60 dakika):**

1. **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU → Kaydet.**
2. **Çalışma zamanı → Tümünü çalıştır.** ("Yine de çalıştır" uyarısını onaylayın.)
3. Drive izni isterse verin. Veriler bir önceki defterin Drive'a kaydettiği `faz4/data` klasöründen okunur; **dosya yüklemeniz gerekmez.**
4. Bitince `faz4_llm_results.zip` iner; deponun ilgili klasörüne açın (bkz. REPRODUCE.md). Bağlantı koparsa yeniden **Tümünü çalıştır** deyin, kaldığı yerden devam eder.

Model: Qwen3-8B (4-bit). Yüklenemezse otomatik olarak Qwen2.5-7B-Instruct kullanılır; hangisinin kullanıldığı sonuç dosyasına yazılır."""

SETUP = """!pip -q install -U transformers accelerate bitsandbytes
import torch, os; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK — T4 GPU seçin')
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/faz4'; DATA = OUT + '/data'; RUNS = OUT + '/llm'
os.makedirs(RUNS, exist_ok=True)
print(sorted(os.listdir(DATA)))"""

RUN = r"""import json, time, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

CANDIDATES = ['Qwen/Qwen3-8B', 'Qwen/Qwen2.5-7B-Instruct']
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
for MODEL in CANDIDATES:
    try:
        tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map='auto'); model.eval()
        print('Model yüklendi:', MODEL); break
    except Exception as e:
        print('Yüklenemedi:', MODEL, repr(e)[:200])

OPTS = {3: [('A', 'positive', 'Olumlu'), ('B', 'negative', 'Olumsuz'), ('C', 'neutral', 'Nötr (belirgin olumlu ya da olumsuz değil)')],
        2: [('A', 'positive', 'Olumlu'), ('B', 'negative', 'Olumsuz')]}

def prompt(text, n):
    opts = '\n'.join(f'{k}) {v}' for k, _, v in OPTS[n])
    msg = (f'Aşağıdaki kullanıcı yorumunun genel duygu yönünü belirle.\n\nYorum: {str(text)[:1500]}\n\n'
           f'Seçenekler:\n{opts}\n\nYalnızca seçeneğin harfini yaz.')
    kw = {'enable_thinking': False} if 'Qwen3' in MODEL else {}
    return tok.apply_chat_template([{'role': 'user', 'content': msg}], tokenize=False, add_generation_prompt=True, **kw)

letter_ids = {k: tok.encode(k, add_special_tokens=False)[0] for k in 'ABC'}

@torch.no_grad()
def classify(texts, n, bs=16):
    keys = [k for k, _, _ in OPTS[n]]; labs = [l for _, l, _ in OPTS[n]]
    ids = torch.tensor([letter_ids[k] for k in keys]).to(model.device); out = []
    for i in range(0, len(texts), bs):
        enc = tok([prompt(t, n) for t in texts[i:i+bs]], return_tensors='pt', padding=True).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        out += [labs[j] for j in logits[:, ids].argmax(-1).tolist()]
    return out

for d in ['TMPR', 'TMRD', 'TRSAv1', 'winvoker']:
    n = 2 if d in ('TMPR', 'TMRD') else 3
    for part in ['gold', 'test']:
        tag = f'{d}_{part}'
        if os.path.exists(f'{RUNS}/{tag}.parquet'):
            print(tag, 'zaten tamam'); continue
        t0 = time.time()
        if part == 'gold':
            df = pd.read_parquet(f'{DATA}/gold_{d}.parquet').reset_index().rename(columns={'index': 'idx'})
        else:
            x = pd.read_parquet(f'{DATA}/{d}_s0.parquet')
            te = x[x.part == 'test']
            df = te.sample(n=min(2000, len(te)), random_state=0).reset_index().rename(columns={'index': 'idx'})
        df['pred'] = classify(df.text.tolist(), n)
        df[['idx', 'pred']].assign(model=MODEL).to_parquet(f'{RUNS}/{tag}.parquet', index=False)
        print(f'{tag} tamam — {len(df)} örnek, {round((time.time()-t0)/60,1)} dk')"""

DOWNLOAD = """import shutil
shutil.make_archive('/content/faz4_llm_results', 'zip', RUNS)
from google.colab import files; files.download('/content/faz4_llm_results.zip')"""

cells = [("markdown", INTRO), ("code", SETUP), ("code", RUN), ("code", DOWNLOAD)]
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [dict(cell_type=k, metadata={}, source=v, **({"execution_count": None, "outputs": []} if k == "code" else {}))
                for k, v in cells]}
json.dump(nb, open("share/Faz4_LLM.ipynb", "w"), ensure_ascii=False, indent=1)
for k, v in cells:
    if k == "code":
        compile(v.replace("!pip", "#pip"), "cell", "exec")
print("ok")
