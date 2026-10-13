"""Çapraz kirlilik, uzunluk kontrolü: winvoker (100k, s0) TF-IDF modeli TMRD testinde; çakışan vs çakışmayan, uzunluk çeyrekleri içinde."""
import json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from auditkit.models import tfidf_lr
x = pd.read_parquet("out/splits/winvoker_s0.parquet"); tr0 = x[x.part == "train"]
tr = tr0  # tam eğitim: alt örneklemde çakışan metinlerin çoğu eğitime girmez
m = tfidf_lr().fit(tr.text, tr.label)
tm = pd.read_parquet("out/tmrd_test_overlap.parquet")
tm["pred"] = m.predict(tm.text); tm = tm[tm.pred != "neutral"].copy()
tm["correct"] = (tm.pred == tm.label).astype(int); tm["q"] = pd.qcut(tm.n_words, 4, labels=False, duplicates="drop")
tab = tm.groupby(["q", "in_wv"]).correct.agg(["mean", "size"]).unstack()
tm["in_wv_i"] = tm.in_wv.astype(int); tm["logw"] = np.log1p(tm.n_words)
fit = smf.logit("correct ~ in_wv_i + logw", tm).fit(disp=0)
res = {"by_quartile": {str(k): v for k, v in tab.round(4).to_dict().items()}, "logit_coef_in_wv": float(fit.params["in_wv_i"]),
       "logit_p_in_wv": float(fit.pvalues["in_wv_i"]), "odds_ratio": float(np.exp(fit.params["in_wv_i"])),
       "acc_overlap": float(tm[tm.in_wv].correct.mean()), "acc_nonoverlap": float(tm[~tm.in_wv].correct.mean()),
       "median_words": tm.groupby("in_wv").n_words.median().to_dict()}
json.dump(res, open("out/cross_len.json", "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str)); print(tab)
