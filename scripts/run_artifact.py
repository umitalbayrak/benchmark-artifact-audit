"""Faz 2.6: winvoker kaynak artefaktı testi (H3).

A) Kaynak etiketi belirliyor mu? Kaynak başına çoğunluk etiketiyle test doğruluğu.
B) Metin türü (Wikipedia/rastgele vs. yorum) ne kadar kolay ayrılıyor?
C) winvoker ile eğitilen duygu modeli gerçek nötr YORUMLARDA (TRSAv1 nötr) nötrü bulabiliyor mu?
   Referans: aynı model mimarisi TRSAv1'in kendi eğitim bölümüyle eğitildiğinde.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from auditkit.loaders import load_winvoker, load_trsav1
from auditkit.models import tfidf_lr

out = {}
w = load_winvoker()
tr, te = w[w.split == "train"], w[w.split == "test"]

# A) kaynak -> etiket
maj = tr.groupby("source").label.agg(lambda s: s.value_counts().index[0])
out["A_source_majority_label_test_acc"] = accuracy_score(te.label, te.source.map(maj))
out["A_neutral_from_wiki_or_random_pct"] = 100 * w[w.label == "neutral"].source.isin(["wiki", "random"]).mean()

# B) metin türü sınıflandırıcısı
istype = lambda d: d.source.isin(["wiki", "random"]).astype(int)
mB = tfidf_lr().fit(tr.text, istype(tr))
out["B_texttype_test_acc"] = accuracy_score(istype(te), mB.predict(te.text))

# C) duygu modeli
mC = tfidf_lr().fit(tr.text, tr.label)
pin = mC.predict(te.text)
rep_in = classification_report(te.label, pin, output_dict=True, digits=4)
out["C_in_dist_macroF1"] = rep_in["macro avg"]["f1-score"]
out["C_in_dist_neutral_recall"] = rep_in["neutral"]["recall"]

t = load_trsav1()
t_tr, t_te = train_test_split(t, test_size=0.2, stratify=t.label, random_state=0)
pout = mC.predict(t_te.text)
ct = pd.crosstab(t_te.label, pout, normalize="index").round(4)
out["C_trsav1_neutral_recall_winvokermodel"] = float((pout[t_te.label.values == "neutral"] == "neutral").mean())
out["C_trsav1_pred_dist_for_neutral_reviews"] = ct.loc["neutral"].to_dict()
out["C_trsav1_macroF1_winvokermodel"] = classification_report(t_te.label, pout, output_dict=True)["macro avg"]["f1-score"]

mRef = tfidf_lr().fit(t_tr.text, t_tr.label)
pref = mRef.predict(t_te.text)
out["C_ref_trsav1_neutral_recall_inDomain"] = float((pref[t_te.label.values == "neutral"] == "neutral").mean())
out["C_ref_trsav1_macroF1_inDomain"] = classification_report(t_te.label, pref, output_dict=True)["macro avg"]["f1-score"]

# Örnekler: winvoker modelinin TRSAv1 nötr yorumlarına verdiği etiketler
ex = t_te[t_te.label == "neutral"].assign(pred=pout[t_te.label.values == "neutral"]).sample(15, random_state=1)
ex[["text", "pred"]].to_csv("out/artifact_examples.csv", index=False)

json.dump(out, open("out/artifact_h3.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
print(ct)
