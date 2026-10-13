"""Ortak klasik model: hashing TF-IDF (kelime 1-2 gram + karakter 3-5 gram) + Lojistik Regresyon.

HashingVectorizer kelime dağarcığı tutmadığı için 500 bin belgede bellek dostudur (7 GB ortamda çalışır).
"""
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union


def tfidf_lr(C=4.0, n_features=2**20, seed=0):
    feats = make_union(
        make_pipeline(HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=n_features,
                                        alternate_sign=False, norm=None, dtype=np.float32),
                      TfidfTransformer(sublinear_tf=True)),
        make_pipeline(HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=n_features,
                                        alternate_sign=False, norm=None, dtype=np.float32),
                      TfidfTransformer(sublinear_tf=True)),
    )
    clf = LogisticRegression(C=C, max_iter=300, random_state=seed)
    return make_pipeline(feats, clf)
