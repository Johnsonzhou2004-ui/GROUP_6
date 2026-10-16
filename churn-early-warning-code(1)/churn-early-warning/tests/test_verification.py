"""Tests for the AI-output verifier and the pipeline's data checks.  Run:  python -m pytest -q"""
import sqlite3

import pandas as pd
import pytest

from src.config import DB_PATH
from src.genai import customer_facts, generate_retention_script, select_offers, verify_script

ROW = pd.Series({
    "contract": "Month-to-month", "payment_method": "Electronic check", "internet_service": "Fiber optic",
    "tech_support": "No", "online_security": "No", "tenure_months": 8, "monthly_charges": 85.70,
})


def test_offer_targeting_matches_customer():
    assert select_offers(ROW) == ["contract_upgrade", "autopay"]


def test_template_script_passes_verifier():
    out = generate_retention_script(ROW)
    assert out["verified"], out["issues"]


@pytest.mark.parametrize("bad, expected", [
    ("We will give you a 50% discount if you stay with us today, would you like that?", "percentage"),
    ("Your bill is USD 40 a month and you are likely to leave, so we offer a discount. Okay?", "forbidden"),
    ("As a senior customer we can give you a free plan review for 6 months, shall we proceed with it today?", "forbidden"),
])
def test_verifier_catches_hallucinations(bad, expected):
    facts = customer_facts(ROW)
    ver = verify_script(bad, facts)
    assert not ver.passed
    assert any(expected in i.lower() for i in ver.issues)


def test_verifier_requires_call_to_action():
    facts = customer_facts(ROW)
    text = " ".join(["word"] * 80) + " discount."
    assert not verify_script(text, facts).passed


def test_database_integrity():
    con = sqlite3.connect(DB_PATH)
    n_c = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    n_s = con.execute("SELECT COUNT(*) FROM churn_scores").fetchone()[0]
    orphans = con.execute("SELECT COUNT(*) FROM billing b LEFT JOIN customers c USING(customer_id) "
                          "WHERE c.customer_id IS NULL").fetchone()[0]
    bad_prob = con.execute("SELECT COUNT(*) FROM churn_scores WHERE churn_prob<0 OR churn_prob>1").fetchone()[0]
    assert n_c == n_s == 7043 and orphans == 0 and bad_prob == 0
