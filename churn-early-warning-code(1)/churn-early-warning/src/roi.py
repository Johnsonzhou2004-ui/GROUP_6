"""Retention campaign ROI logic (shared by the dashboard and the report).

Expected value of contacting a customer with churn probability p and monthly charge m:
    benefit = p * save_rate * m * months_value          (revenue kept if they would have left and are saved)
    cost    = contact_cost + p * save_rate * m * discount_pct * months_value   (discount only paid on saves)
Contact a customer only if benefit > cost.  Campaign ROI = (benefit - cost) / cost.
"""
import pandas as pd


def campaign_roi(df: pd.DataFrame, threshold: float, contact_cost: float, save_rate: float,
                 months_value: int, discount_cost_pct: float) -> dict:
    target = df[df["churn_prob"] >= threshold]
    n = len(target)
    if n == 0:
        return {"contacted": 0, "expected_churners": 0.0, "expected_saves": 0.0, "revenue_saved": 0.0,
                "cost": 0.0, "net_benefit": 0.0, "roi_pct": 0.0}
    exp_churners = float(target["churn_prob"].sum())
    saves = exp_churners * save_rate
    revenue_saved = float((target["churn_prob"] * save_rate * target["monthly_charges"] * months_value).sum())
    discount = revenue_saved * discount_cost_pct
    cost = n * contact_cost + discount
    net = revenue_saved - cost
    return {
        "contacted": int(n),
        "expected_churners": round(exp_churners, 1),
        "expected_saves": round(saves, 1),
        "revenue_saved": round(revenue_saved, 0),
        "cost": round(cost, 0),
        "net_benefit": round(net, 0),
        "roi_pct": round(100 * net / cost, 0) if cost else 0.0,
    }


def sweep_thresholds(df: pd.DataFrame, thresholds, **kw) -> pd.DataFrame:
    rows = [{"threshold": t, **campaign_roi(df, t, **kw)} for t in thresholds]
    return pd.DataFrame(rows)
