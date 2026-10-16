"""Customer Churn Early-Warning Dashboard (Streamlit).

Run:  streamlit run app/dashboard.py
Tabs: 1) Overview  2) At-risk call list  3) Customer detail + GenAI script  4) Retention ROI simulator
All numbers come from the SQLite database via SQL queries (src/db.py + sql/02_queries.sql).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DB_PATH, ROI_DEFAULTS
from src.db import read_sql, run_query
from src.genai import generate_retention_script
from src.roi import campaign_roi, sweep_thresholds

st.set_page_config(page_title="Churn Early-Warning", page_icon=":bar_chart:", layout="wide")

NAVY, TEAL, CORAL = "#12355B", "#1B998B", "#E4572E"
TIER_ORDER = ["Critical", "High", "Medium", "Low"]
TIER_COLORS = {"Low": "#9BC4A8", "Medium": "#F2C14E", "High": "#F28C38", "Critical": CORAL}

st.markdown(
    f"""<style>
    .block-container {{padding-top: 1.6rem;}}
    h1, h2, h3 {{color: {NAVY};}}
    div[data-testid="stMetric"] {{background:#F4F7FB; border-left:4px solid {NAVY}; padding:10px 14px; border-radius:6px;}}
    </style>""",
    unsafe_allow_html=True,
)

if not Path(DB_PATH).exists():
    st.error("Database not found. Run `python run_pipeline.py` first.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_all() -> pd.DataFrame:
    return read_sql("SELECT * FROM v_customer_360")


df_all = load_all()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.header("Filters")
contracts = st.sidebar.multiselect("Contract", sorted(df_all["contract"].unique()), default=sorted(df_all["contract"].unique()))
internet = st.sidebar.multiselect("Internet service", sorted(df_all["internet_service"].unique()),
                                  default=sorted(df_all["internet_service"].unique()))
bands = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
tenure_sel = st.sidebar.multiselect("Tenure band", bands, default=bands)
active_only = st.sidebar.checkbox("Active customers only (call-list view)", value=True,
                                  help="Customers who have not churned yet. Untick to back-test on all customers.")

f = df_all[df_all["contract"].isin(contracts) & df_all["internet_service"].isin(internet)
           & df_all["tenure_band"].isin(tenure_sel)]
if active_only:
    f = f[f["churn"] == 0]

st.title("Customer Churn Early-Warning Dashboard")
st.caption("Telco Customer Churn dataset (7,043 customers) | SQL database -> Python model -> dashboard -> GenAI retention scripts")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "At-risk call list", "Customer detail & GenAI script", "Retention ROI simulator"])

# ------------------------------------------------------------------ tab 1: overview
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers in view", f"{len(f):,}")
    c2.metric("Avg predicted churn risk", f"{f['churn_prob'].mean():.1%}" if len(f) else "-")
    n_hi = int(f["risk_tier"].isin(["High", "Critical"]).sum())
    c3.metric("High + Critical customers", f"{n_hi:,}")
    exp_loss = float((f["churn_prob"] * f["monthly_charges"]).sum())
    c4.metric("Expected monthly revenue at risk", f"USD {exp_loss:,.0f}")

    left, right = st.columns(2)
    with left:
        tier_counts = f["risk_tier"].value_counts().reindex(TIER_ORDER).fillna(0).reset_index()
        tier_counts.columns = ["risk_tier", "customers"]
        fig = px.bar(tier_counts, x="risk_tier", y="customers", color="risk_tier", color_discrete_map=TIER_COLORS,
                     title="Customers by risk tier", text="customers")
        fig.update_layout(showlegend=False, height=360, xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with right:
        seg = (f.groupby("contract")["churn_prob"].mean().mul(100).round(1).reset_index()
               .rename(columns={"churn_prob": "avg_risk_pct"}))
        fig = px.bar(seg, x="contract", y="avg_risk_pct", title="Average predicted risk by contract (%)",
                     text="avg_risk_pct", color_discrete_sequence=[NAVY])
        fig.update_layout(height=360, xaxis_title=None, yaxis_title="Predicted risk (%)")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Highest-risk segments (SQL: GROUP BY + HAVING)")
    st.dataframe(run_query("high_risk_segments"), width="stretch", hide_index=True)
    with st.expander("Back-test: do the risk tiers match what really happened? (all customers)"):
        st.dataframe(run_query("risk_tier_summary"), width="stretch", hide_index=True)

# ------------------------------------------------------------------ tab 2: call list
with tab2:
    st.subheader("Prioritised call list")
    st.caption("Ranked by expected monthly loss = churn probability x monthly charge. Reasons are the model's top risk-increasing factors.")
    tiers = st.multiselect("Risk tiers", TIER_ORDER, default=["Critical", "High"])
    calls = f[f["risk_tier"].isin(tiers)].copy()
    calls["expected_monthly_loss"] = (calls["churn_prob"] * calls["monthly_charges"]).round(2)
    calls["churn_prob"] = calls["churn_prob"].round(3)
    calls["top_drivers"] = calls["top_drivers"].fillna("").str.replace("|", "; ", regex=False)
    calls = calls.sort_values("expected_monthly_loss", ascending=False)
    show = calls[["customer_id", "risk_tier", "churn_prob", "contract", "internet_service", "tenure_months",
                  "monthly_charges", "expected_monthly_loss", "top_drivers"]]
    st.dataframe(show, width="stretch", hide_index=True, height=440)
    st.download_button("Download call list (CSV)", show.to_csv(index=False).encode(), "call_list.csv", "text/csv")

# ------------------------------------------------------------------ tab 3: customer detail + GenAI
with tab3:
    st.subheader("Customer detail and retention script")
    pool = f.sort_values("churn_prob", ascending=False)
    if pool.empty:
        st.info("No customers match the current filters.")
    else:
        cid = st.selectbox("Customer ID (sorted by risk)", pool["customer_id"].head(300).tolist())
        row = pool[pool["customer_id"] == cid].iloc[0]
        a, b = st.columns([1, 2])
        with a:
            st.metric("Churn probability", f"{row['churn_prob']:.1%}")
            st.markdown(f"**Risk tier:** {row['risk_tier']}")
            st.markdown(f"**Contract:** {row['contract']}  \n**Internet:** {row['internet_service']}  \n"
                        f"**Tenure:** {row['tenure_months']} months  \n**Monthly charge:** USD {row['monthly_charges']:.2f}  \n"
                        f"**Payment:** {row['payment_method']}")
            st.markdown("**Why flagged:**")
            for d in (row["top_drivers"] if pd.notna(row["top_drivers"]) else "").split("|"):
                if d:
                    st.markdown(f"- {d}")
        with b:
            if st.button("Generate retention script", type="primary"):
                st.session_state["gen"] = (cid, generate_retention_script(row))
            gen = st.session_state.get("gen")
            if gen and gen[0] == cid:
                out = gen[1]
                st.text_area("Script (editable)", out["script"], height=220)
                if out["verified"]:
                    st.success("Verification passed: no unapproved offers, amounts or sensitive terms.")
                else:
                    st.error("Needs human review: " + "; ".join(out["issues"]))
                st.caption(f"Source: {out['source']}  |  Facts given to the AI: {out['facts']}")
            else:
                st.caption("The AI may only use the customer's real facts and the pre-approved offers. "
                           "Every script is checked by rules before it is shown.")

# ------------------------------------------------------------------ tab 4: ROI simulator
with tab4:
    st.subheader("Retention campaign ROI simulator")
    st.caption("Applies to ACTIVE customers in the current filter. Assumptions are illustrative - change them to match your business.")
    c1, c2, c3 = st.columns(3)
    thr = c1.slider("Contact customers with churn probability >=", 0.10, 0.90, 0.50, 0.05)
    cost = c1.number_input("Cost per contact (USD)", 0.0, 100.0, ROI_DEFAULTS["contact_cost"], 1.0)
    save = c2.slider("Save rate among contacted churners", 0.05, 0.60, ROI_DEFAULTS["save_rate"], 0.05)
    months = c2.slider("Months of revenue kept per save", 3, 24, ROI_DEFAULTS["months_value"], 1)
    disc = c3.slider("Discount cost (% of revenue kept)", 0.0, 0.30, ROI_DEFAULTS["discount_cost_pct"], 0.05)

    act = f[f["churn"] == 0] if not active_only else f
    kw = dict(contact_cost=cost, save_rate=save, months_value=months, discount_cost_pct=disc)
    r = campaign_roi(act, thr, **kw)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Customers contacted", f"{r['contacted']:,}")
    m2.metric("Expected churners reached", f"{r['expected_churners']:,.0f}")
    m3.metric("Expected saves", f"{r['expected_saves']:,.0f}")
    m4.metric("Net benefit", f"USD {r['net_benefit'] / 1000:,.1f}k")
    m5.metric("ROI", f"{r['roi_pct']:,.0f}%")

    sweep = sweep_thresholds(act, [round(x, 2) for x in [0.1 + 0.05 * i for i in range(17)]], **kw)
    sweep = sweep[sweep["contacted"] > 0]
    fig = px.line(sweep, x="threshold", y="net_benefit", markers=True, title="Net benefit by contact threshold",
                  color_discrete_sequence=[NAVY])
    fig.update_layout(height=360, yaxis_title="Net benefit (USD)", xaxis_title="Threshold")
    st.plotly_chart(fig, width="stretch")
    st.caption("Model scores are out-of-fold predictions. The dataset is one snapshot with no dates, so this is a "
               "decision-support simulation, not a forecast.")
