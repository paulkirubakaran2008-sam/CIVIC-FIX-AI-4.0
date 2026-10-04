"""CivicFix AI v4 core: model training, hybrid severity engine, explainability.

Kept in a separate module (leading underscore => Vercel does not treat it as a route).
"""
import csv
import os
import re
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.joblib")
DATA_PATH = os.path.join(os.path.dirname(HERE), "dataset.csv")
MAX_LEN = 600

# ----------------------------------------------------------------- knowledge
DEPARTMENTS = {
    "Electrical": "Electrical Maintenance",
    "Plumbing": "Plumbing & Water Works",
    "Water Leakage": "Plumbing & Water Works",
    "Toilet/Sanitation": "Housekeeping & Sanitation",
    "Garbage": "Housekeeping & Waste Management",
    "Road/Pathway": "Estate & Civil Works",
    "Pothole": "Estate & Civil Works",
    "Civil": "Estate & Civil Works",
    "Carpentry": "Carpentry & Furniture",
    "Class Items": "Academic Infrastructure",
}
# inherent risk of a category if ignored (0-100)
CATEGORY_RISK = {
    "Electrical": 90, "Water Leakage": 60, "Plumbing": 55, "Toilet/Sanitation": 60,
    "Road/Pathway": 55, "Pothole": 60, "Civil": 60, "Garbage": 40,
    "Carpentry": 40, "Class Items": 25,
}
ACTIONS = {
    "Electrical": "Isolate the circuit / switch off power, cordon off the area, send a licensed electrician.",
    "Plumbing": "Shut the local valve, clear the blockage, inspect pipe joints and drains.",
    "Water Leakage": "Close the supply valve, contain the water, inspect pipe/tank for the leak source.",
    "Toilet/Sanitation": "Dispatch housekeeping for cleaning and disinfection; check drainage and water supply.",
    "Garbage": "Schedule immediate waste pickup and sanitise the spot; review bin capacity.",
    "Road/Pathway": "Barricade the damaged stretch and schedule repair; reroute foot traffic meanwhile.",
    "Pothole": "Mark/barricade the pothole and schedule patching before it widens.",
    "Civil": "Inspect structure for safety, restrict access if needed, plan repair work.",
    "Carpentry": "Remove the unsafe item from use and repair or replace it.",
    "Class Items": "Log the asset, replace or repair the item, confirm with the class in-charge.",
}

# severity evidence: (regex, label, points)
CRITICAL = [
    (r"\bspark(s|ing|ed)?\b", "sparking", 22), (r"short[- ]?circuit", "short circuit", 22),
    (r"exposed (electric(al)? )?wires?|live wire|bare wire", "exposed wiring", 22),
    (r"shock|electrocut", "shock risk", 22), (r"\b(fire|smoke|smoking|burning|burnt)\b", "fire/smoke", 24),
    (r"collaps|caving|about to fall|falling", "collapse/fall risk", 20),
    (r"injur|bleed|hospital|accident|\b(fell|fallen|slipped|skid(ded)?|crash(ed)?|tripped)\b", "injury/accident", 18), (r"gas leak", "gas leak", 24),
    (r"flood", "flooding", 16), (r"sewage", "sewage", 14), (r"overflow", "overflow", 10),
    (r"snake|stray dog|bees|beehive|insect nest", "wildlife hazard", 14),
    (r"unsafe|dangerous|hazard|risk", "safety hazard", 14),
    (r"immediate|urgent|emergency|asap", "marked urgent", 12),
    (r"severe|serious|major|critical|heavily", "severe wording", 10),
    (r"open (manhole|drain)|manhole", "open manhole/drain", 16),
    (r"broken glass|sharp", "sharp object", 12),
]
MODERATE = [
    (r"continuous|non[- ]?stop|keeps? leaking|leaking", "ongoing leak", 6),
    (r"block(ed|age|ing)|clog", "blockage", 6), (r"smell|stink|foul|odou?r", "foul smell", 5),
    (r"broken|damaged|cracked?|loose|not working|stopped working|dead", "damage / malfunction", 4),
    (r"overflowing|burst", "burst/overflow", 6),
    (r"\b(big|large|huge|deep|wide|massive)\b", "large size", 8),
]
MITIGATORS = [
    (r"\b(minor|small|slight(ly)?|little|tiny|cosmetic|few)\b", "described as minor", -12),
    (r"needs? attention|please check|when possible|no hurry", "non-urgent tone", -4),
]
EXPOSURE = [
    (r"student|children|kids|crowd|people", "students nearby", 8),
    (r"hostel|dorm", "hostel", 6), (r"\b(lab|laborator)", "laboratory", 6),
    (r"canteen|cafeteria|kitchen", "food area", 6), (r"stair|ramp|lift|elevator", "stairs/lift", 6),
    (r"main gate|entrance|corridor|walkway|pathway", "high-footfall area", 5),
    (r"auditorium|library|playground|classroom|seminar", "public campus area", 3),
]
CATEGORY_KEYWORDS = {
    "Electrical": r"electric|switch|socket|wire|wiring|power|fan\b|light|bulb|tube|spark|short|circuit|voltage|mcb|current",
    "Plumbing": r"pipe|drain|drainage|tap\b|basin|sink|valve|blockage|clog",
    "Water Leakage": r"leak|drip|seep|dampness|tank|overhead",
    "Toilet/Sanitation": r"toilet|washroom|restroom|bathroom|urinal|flush|sanitation|sewage|latrine",
    "Garbage": r"garbage|trash|waste|dustbin|litter|dump|rubbish|bin\b",
    "Road/Pathway": r"road|path|walkway|pavement|footpath|tiles?|speed breaker",
    "Pothole": r"pothole|pit\b|crater|hole in",
    "Civil": r"wall|ceiling|plaster|paint|crack|roof|pillar|beam|slab|floor",
    "Carpentry": r"door|window|bench|desk|chair|table|wood|furniture|cupboard|hinge|lock|shelf",
    "Class Items": r"projector|board|whiteboard|marker|duster|clock|speaker|mic|screen|remote|computer|device|ac\b",
}


def _norm(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _score_rules(text, rules):
    t, pts, found = text.lower(), 0, []
    for rx, label, p in rules:
        if re.search(rx, t):
            pts += p
            found.append((label, p))
    return pts, found


# ----------------------------------------------------------------- training
def _dedupe_key(s):
    s = s.lower()
    s = re.sub(r"^please report:\s*", "", s)
    s = re.sub(r"\s+and needs attention$", "", s)
    s = re.sub(r"\s+on campus$", "", s)
    return s.strip()


def load_dataset(path=DATA_PATH):
    rows, seen = [], set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text = _norm(r["complaint"])
            k = (_dedupe_key(text), r["category"], r["severity"])
            if k in seen:  # remove templated duplicates -> faster, less biased
                continue
            seen.add(k)
            rows.append((_dedupe_key(text), r["category"].strip(), r["severity"].strip()))
    return rows


def train(path=DATA_PATH):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_union
    from sklearn.svm import LinearSVC

    rows = load_dataset(path)
    X = [r[0] for r in rows]
    vec = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
    )
    Xv = vec.fit_transform(X)
    models = {}
    for name, idx in (("category", 1), ("severity", 2)):
        y = [r[idx] for r in rows]
        models[name] = CalibratedClassifierCV(LinearSVC(C=3), cv=3).fit(Xv, y)
    return {"vec": vec, "category": models["category"], "severity": models["severity"], "n_train": len(rows)}


_BUNDLE = None


def get_bundle():
    """Load the pre-trained artifact; retrain transparently if it is missing/incompatible."""
    global _BUNDLE
    if _BUNDLE is None:
        try:
            import joblib
            _BUNDLE = joblib.load(MODEL_PATH)
        except Exception:
            _BUNDLE = train()
        _BUNDLE["names"] = _BUNDLE["vec"].get_feature_names_out()
        # averaged linear weights of the calibrated SVMs -> used for "why" explanations
        cat = _BUNDLE["category"]
        _BUNDLE["cat_coef"] = np.mean([c.estimator.coef_ for c in cat.calibrated_classifiers_], axis=0)
    return _BUNDLE


# --------------------------------------------------------------- inference
def _explain_terms(b, xrow, cls_index, k=5):
    """Top words/phrases that pushed the model toward the predicted category."""
    contrib = b["cat_coef"][cls_index] * xrow.toarray()[0]
    order = np.argsort(-contrib)
    out = []
    for i in order:
        if contrib[i] <= 0 or len(out) >= k:
            break
        n = b["names"][i]
        if n.startswith("tfidfvectorizer-1__"):  # word n-grams only (readable)
            out.append(n.split("__", 1)[1])
    return out


def _predict_many(texts):
    b = get_bundle()
    X = b["vec"].transform([_dedupe_key(t) for t in texts])
    return b, X, b["category"].predict_proba(X), b["severity"].predict_proba(X)


def analyze_many(texts):
    b, X, cat_p, sev_p = _predict_many(texts)
    cat_cls, sev_cls = list(b["category"].classes_), list(b["severity"].classes_)
    results = []
    for i, text in enumerate(texts):
        cp, sp = cat_p[i], sev_p[i]
        order = np.argsort(-cp)
        ci = int(order[0])
        category, cat_conf = cat_cls[ci], float(cp[ci])
        method = "ML"
        # keyword rescue when the model is unsure
        if cat_conf < 0.40:
            votes = {c: len(re.findall(rx, text.lower())) for c, rx in CATEGORY_KEYWORDS.items()}
            best = max(votes, key=votes.get)
            if votes[best] > 0:
                category, method = best, "ML + keyword rescue"
                ci = cat_cls.index(best)
        margin = float(cp[order[0]] - cp[order[1]])

        # ---- hybrid severity: ML expectation + evidence rules
        w = {"High": 100, "Medium": 55, "Low": 8}
        base = sum(float(sp[j]) * w[c] for j, c in enumerate(sev_cls))
        crit, crit_f = _score_rules(text, CRITICAL)
        mod, mod_f = _score_rules(text, MODERATE)
        mit, mit_f = _score_rules(text, MITIGATORS)
        expo, expo_f = _score_rules(text, EXPOSURE)
        hybrid = base * 0.6 + min(crit, 45) + min(mod, 10) + max(mit, -20) + min(expo, 12) * 0.5
        if any(pt >= 20 for _, pt in crit_f):  # safety floor: life-safety evidence can never be outvoted by ML
            hybrid = max(hybrid, 72.0)
        hybrid = float(np.clip(hybrid, 0, 100))
        severity = "High" if hybrid >= 68 else "Medium" if hybrid >= 38 else "Low"
        sj = int(np.argmax(sp))

        # ---- priority
        exposure = min(expo * 8, 100)
        pscore = float(np.clip(0.68 * hybrid + 0.20 * CATEGORY_RISK.get(category, 50) + 0.12 * exposure, 0, 100))
        if pscore >= 70: priority, sla = "P1 - Critical", "Respond within 2 hours"
        elif pscore >= 55: priority, sla = "P2 - High", "Respond within 8 hours"
        elif pscore >= 38: priority, sla = "P3 - Medium", "Resolve within 2 days"
        else: priority, sla = "P4 - Low", "Resolve within 7 days"

        reasons = [f"{l} (+{p})" for l, p in crit_f] + [f"{l} (+{p})" for l, p in mod_f] \
            + [f"{l} ({p})" for l, p in mit_f] + [f"{l} (exposure)" for l, _ in expo_f]
        results.append({
            "complaint": text,
            "category": category,
            "category_confidence": round(cat_conf * 100, 1),
            "category_method": method,
            "category_probabilities": {c: round(float(cp[j]) * 100, 1) for j, c in enumerate(cat_cls)},
            "alternatives": [{"category": cat_cls[int(j)], "confidence": round(float(cp[j]) * 100, 1)} for j in order[1:3]],
            "needs_review": bool(cat_conf < 0.45 or margin < 0.12),
            "severity": severity,
            "severity_confidence": round(float(sp[sj]) * 100, 1),
            "ml_severity": sev_cls[sj],
            "ml_severity_score": round(base, 1),
            "hybrid_severity_score": round(hybrid, 1),
            "severity_reasons": reasons or ["no strong risk signals - ML estimate only"],
            "priority": priority,
            "priority_score": round(pscore, 1),
            "sla": sla,
            "escalate": bool(pscore >= 70),
            "department": DEPARTMENTS.get(category, "Estate & Civil Works"),
            "suggested_action": ACTIONS.get(category, "Inspect and assign."),
            "key_terms": _explain_terms(b, X[i], ci),
        })
    return results


@lru_cache(maxsize=2048)
def _cached(text):
    return analyze_many([text])[0]


def analyze_one(text):
    return dict(_cached(text))


def validate(text):
    text = _norm(text or "")
    if len(text) < 3:
        raise ValueError("Complaint is too short.")
    return text[:MAX_LEN]
