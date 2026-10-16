"""Churn model: train, evaluate honestly, score every customer, explain each score.

Design choices (explained in docs/report):
  * Logistic regression is the production model: transparent, well calibrated, and each
    prediction can be decomposed into plain-language "reasons" for the call-centre agent.
  * Gradient boosting is trained as a challenger. We only switch if it is clearly better.
  * gender is excluded from the features (no business reason to use it, and it carries no signal).
  * Scores written to the database are OUT-OF-FOLD (5-fold) predictions, so every customer is
    scored by a model that never saw that customer. This keeps the risk-tier table honest.

Run:  python -m src.model
"""
import json
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DB_PATH, OUT_DIR, RANDOM_STATE, RISK_TIERS, TEST_SIZE

# total_charges is deliberately NOT a feature: it is ~ tenure x monthly_charges, so it is redundant and
# makes coefficients hard to interpret (collinearity). Dropping it costs no accuracy (see README).
NUMERIC = ["tenure_months", "monthly_charges"]
CATEGORICAL = ["senior_citizen", "partner", "dependents", "phone_service", "multiple_lines",
               "internet_service", "online_security", "online_backup", "device_protection",
               "tech_support", "streaming_tv", "streaming_movies", "contract",
               "paperless_billing", "payment_method"]
FEATURES = NUMERIC + CATEGORICAL

# Only behavioural / contractual drivers are shown to agents. Demographic traits are used by the
# model but never surfaced as a "reason" and never allowed into customer-facing scripts.
NON_ACTIONABLE = {"senior_citizen", "partner", "dependents"}

READABLE = {
    ("contract", "Month-to-month"): "Month-to-month contract",
    ("contract", "One year"): "One-year contract (not yet on two-year)",
    ("internet_service", "Fiber optic"): "Fiber optic plan",
    ("payment_method", "Electronic check"): "Pays by electronic check",
    ("paperless_billing", "Yes"): "Paperless billing",
    ("online_security", "No"): "No Online Security add-on",
    ("tech_support", "No"): "No Tech Support add-on",
    ("online_backup", "No"): "No Online Backup add-on",
    ("device_protection", "No"): "No Device Protection add-on",
    ("streaming_tv", "Yes"): "Streams TV (higher bill)",
    ("streaming_movies", "Yes"): "Streams movies (higher bill)",
    ("multiple_lines", "Yes"): "Multiple phone lines",
}


def load_features(db_path=DB_PATH) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM v_customer_360", con)
    con.close()
    df["senior_citizen"] = df["senior_citizen"].map({0: "No", 1: "Yes"})
    # new customers have no billing history yet -> 0 is the right "total paid so far"
    df["total_charges"] = df["total_charges"].fillna(0.0)
    return df


def make_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])


def tier_of(p: float) -> str:
    for upper, label in RISK_TIERS:
        if p < upper:
            return label
    return RISK_TIERS[-1][1]


def explain(pipe: Pipeline, X: pd.DataFrame, top_k: int = 3) -> list[str]:
    """Per-customer top risk-increasing reasons from the logistic model.
    contribution_j = coef_j * (x_j - mean_j): effect versus the average customer."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Z = pre.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    contrib = clf.coef_[0] * (Z - Z.mean(axis=0))
    names = pre.get_feature_names_out()

    def label(col: str):
        kind, rest = col.split("__", 1)
        if kind == "num":
            return rest, None
        for feat in CATEGORICAL:
            if rest.startswith(feat + "_"):
                return feat, rest[len(feat) + 1:]
        return rest, None

    meta = [label(n) for n in names]
    out = []
    for row in contrib:
        reasons = []
        for j in np.argsort(-row):
            if row[j] <= 0.05 or len(reasons) == top_k:
                break
            feat, val = meta[j]
            if feat in NON_ACTIONABLE:
                continue
            if feat == "tenure_months":
                text = "Short tenure (new customer)"
            elif feat == "monthly_charges":
                text = "High monthly bill"
            else:
                text = READABLE.get((feat, val))
                if text is None:
                    continue
            reasons.append(text)
        out.append("|".join(reasons))
    return out


def top_decile_lift(y, p) -> float:
    n = max(1, len(y) // 10)
    idx = np.argsort(-p)[:n]
    return float(np.mean(np.asarray(y)[idx]) / np.mean(y))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    X, y = df[FEATURES], df["churn"].values

    # ---------- 1) honest hold-out comparison: logistic vs gradient boosting ----------
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    candidates = {
        "Logistic regression": LogisticRegression(max_iter=2000, C=0.5),
        "Gradient boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                            max_iter=200, random_state=RANDOM_STATE),
    }
    results = {}
    for name, est in candidates.items():
        pipe = make_pipeline(est).fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        results[name] = {
            "roc_auc": round(roc_auc_score(yte, p), 4),
            "pr_auc": round(average_precision_score(yte, p), 4),
            "brier": round(brier_score_loss(yte, p), 4),
            "precision_at_0.5": round(precision_score(yte, pred), 4),
            "recall_at_0.5": round(recall_score(yte, pred), 4),
            "top_decile_lift": round(top_decile_lift(yte, p), 3),
        }
    results["baseline_churn_rate_test"] = round(float(yte.mean()), 4)
    results["n_train"], results["n_test"] = int(len(ytr)), int(len(yte))

    # ---------- 2) production model: logistic regression (interpretable) ----------
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(make_pipeline(candidates["Logistic regression"]), X, y, cv=cv,
                            method="predict_proba")[:, 1]
    results["oof_roc_auc_all_customers"] = round(roc_auc_score(y, oof), 4)
    results["oof_top_decile_lift"] = round(top_decile_lift(y, oof), 3)

    final = make_pipeline(LogisticRegression(max_iter=2000, C=0.5)).fit(X, y)
    joblib.dump(final, OUT_DIR / "churn_model.joblib")

    # coefficient table for the report (top positive / negative effects)
    names = final.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(final.named_steps["clf"].coef_[0], index=names).sort_values()
    coefs.to_csv(OUT_DIR / "model_coefficients.csv", header=["coef"])

    # ---------- 3) write scores back to SQL ----------
    drivers = explain(final, X)
    now = datetime.now().isoformat(timespec="seconds")
    rows = [(cid, float(p), tier_of(p), d, now) for cid, p, d in zip(df["customer_id"], oof, drivers)]
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM churn_scores")
    con.executemany("INSERT INTO churn_scores VALUES (?,?,?,?,?)", rows)
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM churn_scores").fetchone()[0]
    assert n == len(df), "Score table must cover every customer"
    con.close()

    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"[model] wrote {n:,} scores to churn_scores")


if __name__ == "__main__":
    main()
