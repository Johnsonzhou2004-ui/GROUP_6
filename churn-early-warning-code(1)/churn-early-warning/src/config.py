"""Central configuration: paths, risk tiers and business-simulation defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DB_PATH = ROOT / "data" / "churn.db"
SQL_DIR = ROOT / "sql"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Risk tiers are defined on predicted churn probability.
# (upper bound exclusive, label) - evaluated in order.
RISK_TIERS = [
    (0.25, "Low"),
    (0.50, "Medium"),
    (0.75, "High"),
    (1.01, "Critical"),
]

# Default assumptions for the retention ROI simulator (all editable in the dashboard).
ROI_DEFAULTS = {
    "contact_cost": 8.0,        # cost of one retention call / offer handling (USD)
    "save_rate": 0.30,          # share of contacted churners who are retained
    "months_value": 12,         # months of revenue kept if customer is saved
    "discount_cost_pct": 0.10,  # discount given to saved customer, % of monthly charge
}

# Offers the GenAI assistant is ALLOWED to mention. Anything else is rejected by the verifier.
APPROVED_OFFERS = {
    "contract_upgrade": "a 10% discount for 12 months when you move to a 1-year contract",
    "autopay": "a waived paper-bill fee plus a one-off USD 5 credit for switching to automatic payment",
    "tech_support": "3 months of free Tech Support",
    "online_security": "3 months of free Online Security",
    "loyalty_review": "a free plan review to find a cheaper bundle",
}
