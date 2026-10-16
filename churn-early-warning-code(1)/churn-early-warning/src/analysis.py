"""Run every named SQL query, save the tables, and draw the figures used in the report / slides.

Run:  python -m src.analysis
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIG_DIR, OUT_DIR, ROI_DEFAULTS
from src.db import query_names, read_sql, run_query
from src.roi import campaign_roi, sweep_thresholds

NAVY, TEAL, CORAL, GREY, LIGHT = "#12355B", "#1B998B", "#E4572E", "#8A94A6", "#E9EEF5"
TIER_COLORS = {"Low": "#9BC4A8", "Medium": "#F2C14E", "High": "#F28C38", "Critical": CORAL}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": "#333", "xtick.color": "#333", "ytick.color": "#333",
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bar_churn(df, cat, title, name, avg, color=NAVY, rot=0):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.bar(df[cat], df["churn_rate_pct"], color=color, width=0.6)
    ax.axhline(avg, color=CORAL, ls="--", lw=1.3)
    ax.text(len(df) - 0.5, avg + 1, f"Portfolio average {avg:.1f}%", color=CORAL, ha="right", fontsize=9.5)
    for b, v in zip(bars, df["churn_rate_pct"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Churn rate (%)")
    ax.set_ylim(0, max(df["churn_rate_pct"]) * 1.2)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=NAVY)
    plt.setp(ax.get_xticklabels(), rotation=rot, ha="right" if rot else "center")
    save(fig, name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sql_results").mkdir(exist_ok=True)
    res = {n: run_query(n) for n in query_names()}
    for n, d in res.items():
        d.to_csv(OUT_DIR / "sql_results" / f"{n}.csv", index=False)

    kpi = res["portfolio_kpis"].iloc[0]
    avg = float(kpi["churn_rate_pct"])

    # ---- descriptive charts ----
    bar_churn(res["churn_by_contract"], "contract", "Churn rate by contract type", "churn_by_contract", avg, NAVY)
    bar_churn(res["churn_by_tenure_band"], "tenure_band", "Churn rate by customer tenure", "churn_by_tenure", avg, TEAL)
    bar_churn(res["churn_by_internet_service"], "internet_service", "Churn rate by internet service",
              "churn_by_internet", avg, NAVY)
    pm = res["churn_by_payment_method"].copy()
    pm["payment_method"] = pm["payment_method"].str.replace(" (automatic)", "\n(automatic)", regex=False)
    bar_churn(pm, "payment_method", "Churn rate by payment method", "churn_by_payment", avg, TEAL)

    # ---- risk tier charts ----
    tiers = res["risk_tier_summary"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = [TIER_COLORS[t] for t in tiers["risk_tier"]]
    bars = ax.bar(tiers["risk_tier"], tiers["actual_churn_rate_pct"], color=colors, width=0.6)
    for b, v, c in zip(bars, tiers["actual_churn_rate_pct"], tiers["customers"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 2, f"n={c:,}", ha="center", color="white", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Actual churn rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model risk tiers separate churners well", loc="left", fontsize=12, fontweight="bold", color=NAVY)
    save(fig, "risk_tier_actual_churn")

    # ---- heatmap contract x tenure ----
    heat = read_sql("""SELECT contract, tenure_band, ROUND(100.0*AVG(churn),1) AS churn_rate_pct, COUNT(*) AS n
                       FROM v_customer_360 GROUP BY contract, tenure_band""")
    cnt = heat.pivot(index="contract", columns="tenure_band", values="n")
    pv = heat.pivot(index="contract", columns="tenure_band", values="churn_rate_pct")
    rows_, cols_ = ["Month-to-month", "One year", "Two year"], ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
    pv, cnt = pv.loc[rows_, cols_], cnt.loc[rows_, cols_]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    im = ax.imshow(pv.values, cmap="YlOrRd", vmin=0, vmax=60, aspect="auto")
    ax.set_xticks(range(pv.shape[1]), pv.columns)
    ax.set_yticks(range(pv.shape[0]), pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"{v:.1f}%\nn={int(cnt.values[i, j]):,}", ha="center", va="center", fontweight="bold",
                    fontsize=9.5, color="white" if v > 35 else "#222")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Churn rate: contract x tenure", loc="left", fontsize=12, fontweight="bold", color=NAVY)
    save(fig, "heatmap_contract_tenure")
    pv.to_csv(OUT_DIR / "sql_results" / "heatmap_contract_tenure.csv")
    cnt.to_csv(OUT_DIR / "sql_results" / "heatmap_contract_tenure_n.csv")

    # ---- ROI on the ACTIVE base only (you cannot save a customer who has already left) ----
    scored = read_sql("""SELECT customer_id, churn_prob, risk_tier, monthly_charges, churn
                         FROM v_customer_360 WHERE churn = 0""")
    kw = {k: ROI_DEFAULTS[k] for k in ("contact_cost", "save_rate", "months_value", "discount_cost_pct")}
    ths = np.round(np.arange(0.10, 0.91, 0.05), 2)
    sweep = sweep_thresholds(scored, ths, **kw)
    sweep.to_csv(OUT_DIR / "sql_results" / "roi_threshold_sweep.csv", index=False)

    tier_rows = []
    for tier in ["Critical", "High", "Medium", "Low"]:
        sub = scored[scored.risk_tier == tier]
        r = campaign_roi(sub, 0.0, **kw)
        tier_rows.append({"risk_tier": tier, **r})
    tier_roi = pd.DataFrame(tier_rows)
    tier_roi.to_csv(OUT_DIR / "sql_results" / "roi_by_tier.csv", index=False)

    OP = 0.5  # recommended operating point = High + Critical tiers
    op_row = sweep.loc[np.isclose(sweep["threshold"], OP)].iloc[0]
    plot = sweep[sweep["contacted"] >= 20]  # hide thresholds where (almost) nobody is left to contact
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(plot["threshold"], plot["net_benefit"] / 1000, color=NAVY, lw=2.4, marker="o", ms=4)
    ax.set_xlabel("Contact threshold (predicted churn probability)")
    ax.set_ylabel("Expected net benefit (USD '000)", color=NAVY)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(plot["threshold"], plot["roi_pct"], color=TEAL, lw=2.0, ls="--")
    ax2.set_ylim(0, 800)
    ax2.set_ylabel("ROI (%)", color=TEAL)
    ax.axvline(OP, color=CORAL, lw=1.2)
    ax.text(OP - 0.012, ax.get_ylim()[1] * 0.08, "Operating point 0.50\n(High + Critical tiers)",
            color=CORAL, fontsize=9.5, fontweight="bold", va="bottom", ha="right")
    ax.set_title("Net benefit (solid) vs ROI (dashed) by contact threshold",
                 loc="left", fontsize=11.5, fontweight="bold", color=NAVY)
    save(fig, "roi_threshold")

    # ---- drivers chart (model coefficients, readable) ----
    co = pd.read_csv(OUT_DIR / "model_coefficients.csv", index_col=0)["coef"]
    keep = {
        "cat__contract_Month-to-month": "Month-to-month contract", "cat__contract_Two year": "Two-year contract",
        "num__tenure_months": "Tenure (per +1 SD)", "cat__internet_service_Fiber optic": "Fiber optic plan",
        "cat__payment_method_Electronic check": "Electronic check payment",
        "cat__online_security_No": "No Online Security", "cat__tech_support_No": "No Tech Support",
        "num__monthly_charges": "Monthly charges (per +1 SD)",
    }
    sub = pd.Series({v: co[k] for k, v in keep.items() if k in co.index}).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(sub.index, sub.values, color=[TEAL if v < 0 else CORAL for v in sub.values], height=0.65)
    ax.axvline(0, color="#333", lw=0.8)
    for y, v in enumerate(sub.values):
        ax.text(v + (0.03 if v >= 0 else -0.03), y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9.5)
    ax.set_xlabel("Logistic regression coefficient (log-odds)")
    ax.set_title("What drives churn risk", loc="left", fontsize=12, fontweight="bold", color=NAVY)
    ax.set_xlim(sub.min() - 0.45, sub.max() + 0.45)
    save(fig, "model_drivers")
    sub.to_csv(OUT_DIR / "sql_results" / "driver_coefficients_selected.csv", header=["coef"])

    # ---- summary json used by report / slides (single source of truth for every number) ----
    metrics = json.loads((OUT_DIR / "metrics.json").read_text())
    allsc = read_sql("SELECT churn_prob, churn FROM v_customer_360")
    flagged = allsc[allsc.churn_prob >= OP]
    active_tier = res["active_base_by_tier"]
    summary = {
        "kpis": {k: (float(v) if not isinstance(v, str) else v) for k, v in kpi.items()},
        "backtest_flagged_customers": int(len(flagged)),
        "backtest_flagged_pct_base": round(100 * len(flagged) / len(allsc), 1),
        "backtest_flagged_precision_pct": round(100 * float(flagged["churn"].mean()), 1),
        "backtest_churners_captured_pct": round(100 * float(flagged["churn"].sum() / allsc["churn"].sum()), 1),
        "active_customers": int(len(scored)),
        "active_flagged": int((scored.churn_prob >= OP).sum()),
        "roi_defaults": ROI_DEFAULTS,
        "roi_at_operating_point": {k: (float(v) if not isinstance(v, str) else v) for k, v in op_row.to_dict().items()},
        "roi_best_total": sweep.loc[sweep["net_benefit"].idxmax()].to_dict(),
        "roi_by_tier": tier_roi.to_dict(orient="records"),
        "active_base_by_tier": active_tier.to_dict(orient="records"),
        "metrics": metrics,
    }
    # sensitivity of the operating-point result to the save-rate assumption
    sens = []
    for sr in (0.10, 0.20, 0.30):
        k2 = dict(kw, save_rate=sr)
        sens.append({"save_rate": sr, **campaign_roi(scored, OP, **k2)})
    summary["sensitivity_save_rate"] = sens
    pd.DataFrame(sens).to_csv(OUT_DIR / "sql_results" / "roi_sensitivity.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
