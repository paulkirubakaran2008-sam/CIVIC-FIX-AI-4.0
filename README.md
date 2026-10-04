# CivicFix AI v4 — Smart Campus Complaint Triage

Classifies a complaint into **category**, **severity**, **priority (P1–P4)**, routes it to a **department**, gives an **SLA** and a **suggested action**, and explains *why*.

## What's new vs v3
| Area | v3 | v4 |
|---|---|---|
| Model | trained on every cold start | **pre-trained `api/model.joblib`** (loads once, then cached) |
| Classifier | Logistic Regression | **Calibrated LinearSVC** on word + char TF-IDF (better accuracy, real probabilities) |
| Data | 1000 rows, 325 duplicates | duplicates removed → 415 unique, no inflated scores |
| Latency | seconds on cold start | ~20 ms per request when warm, LRU cache for repeats |
| Severity | 12 rules | 40+ evidence rules + exposure context (students, hostel, lab, stairs…) |
| Output | category/severity/priority | + SLA, escalation flag, department, action, key words, low-confidence "needs review" |
| API | 1 endpoint | `GET /api` health, `POST /api`, `POST /api/batch` (ranked, ≤200) |
| UI | single form | dark/light theme, priority gauge, batch triage + CSV import/export, history, voice input, XSS-safe |

## Run / retrain
```bash
pip install -r requirements.txt pandas
python train.py          # prints CV metrics and rewrites api/model.joblib
```
**Always retrain with the same `scikit-learn` version pinned in `requirements.txt`** (the API auto-retrains from `dataset.csv` if the artifact can't be loaded, but that is slower).

## Deploy (Vercel)
Push this folder to GitHub → Import in Vercel → Deploy. `vercel.json` bundles the model.

## API
```bash
curl -X POST https://YOUR-APP.vercel.app/api -H "Content-Type: application/json" \
     -d '{"complaint":"Electrical switch is sparking near students"}'

curl -X POST https://YOUR-APP.vercel.app/api/batch -H "Content-Type: application/json" \
     -d '{"complaints":["Tap leaking in lab","Fan not working"]}'
```

## Honest accuracy note
Dataset is template-generated, so cross-validation looks very high (~94% category / ~97% severity).
On genuinely new wording expect lower (~80% category, since *Plumbing* vs *Water Leakage* vs *Toilet* overlap).
The best next upgrade is adding **real student complaints** to `dataset.csv` and re-running `train.py`.
