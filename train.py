"""Run locally:  python train.py   -> writes api/model.joblib and prints honest metrics."""
import os, sys, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))
import _core
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

rows = _core.load_dataset()
X = [r[0] for r in rows]
print(f"{len(rows)} unique labeled complaints (duplicates removed)")
for name, idx in (("category", 1), ("severity", 2)):
    y = np.array([r[idx] for r in rows])
    p = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                 TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
                      CalibratedClassifierCV(LinearSVC(C=3), cv=3))
    pred = cross_val_predict(p, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"{name}: 5-fold acc={accuracy_score(y, pred):.3f} macro-F1={f1_score(y, pred, average='macro'):.3f}")
bundle = _core.train()
joblib.dump(bundle, _core.MODEL_PATH, compress=3)
print("saved", _core.MODEL_PATH, round(os.path.getsize(_core.MODEL_PATH) / 1024), "KB")
