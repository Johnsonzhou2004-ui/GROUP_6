"""GenAI retention-script assistant with a built-in verification layer.

Flow:  customer row -> approved facts + approved offers -> prompt -> LLM (or offline template)
       -> verify_script() -> "Verified" / "Needs review" shown to the agent.

Why a verifier?  LLMs can invent discounts, quote wrong prices or mention sensitive traits.
Every generated script is therefore checked by deterministic rules BEFORE an agent sees it.
The LLM is optional: with no API key the assistant falls back to a template so the whole
workflow still runs offline (and in the classroom demo).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from src.config import APPROVED_OFFERS

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

# Words that must never appear in a customer-facing script.
FORBIDDEN_TERMS = [
    "churn", "probability", "risk score", "likely to leave", "predicted", "algorithm", "model says",
    "senior", "elderly", "old age", "gender", "male", "female", "married", "single", "dependents",
]


# ----------------------------------------------------------------------------------------
# 1) Which facts and offers is the assistant allowed to use?
# ----------------------------------------------------------------------------------------
def select_offers(row) -> list[str]:
    """Rule-based offer targeting: only offers that match the customer's real situation."""
    offers = []
    if row["contract"] == "Month-to-month":
        offers.append("contract_upgrade")
    if row["payment_method"] == "Electronic check":
        offers.append("autopay")
    if row["internet_service"] != "No" and row["tech_support"] == "No":
        offers.append("tech_support")
    if row["internet_service"] != "No" and row["online_security"] == "No":
        offers.append("online_security")
    if not offers:
        offers.append("loyalty_review")
    return offers[:2]


def customer_facts(row) -> dict:
    return {
        "tenure_months": int(row["tenure_months"]),
        "contract": row["contract"],
        "internet_service": row["internet_service"],
        "monthly_charges": round(float(row["monthly_charges"]), 2),
        "offers": [APPROVED_OFFERS[k] for k in select_offers(row)],
    }


def build_prompt(facts: dict) -> str:
    offers = "\n".join(f"- {o}" for o in facts["offers"])
    return f"""You are drafting a short, friendly phone script for a customer-retention agent.

CUSTOMER FACTS (the only facts you may use):
- Time with us: {facts['tenure_months']} months
- Contract: {facts['contract']}
- Internet plan: {facts['internet_service']}
- Current monthly bill: USD {facts['monthly_charges']}

APPROVED OFFERS (mention these and nothing else; do not change numbers):
{offers}

RULES
1. 80-130 words, plain spoken English, warm and respectful, no pressure.
2. Never mention churn, risk, probability, scores, models or predictions.
3. Never mention age, gender, family status or any personal trait.
4. Do not invent prices, discounts, deadlines or products. Use only the offers above.
5. End with one clear question that invites the customer to take the next step.
Return only the script text."""


# ----------------------------------------------------------------------------------------
# 2) Generation: LLM if available, deterministic template otherwise
# ----------------------------------------------------------------------------------------
def _time_with_us(months: int) -> str:
    if months == 0:
        return "joining us recently"
    if months < 12:
        return f"being with us for {months} month{'s' if months != 1 else ''}"
    if months < 24:
        return "being with us for over a year"
    return f"being with us for {months // 12} years"


def template_script(facts: dict) -> str:
    service = {"DSL": "DSL internet", "Fiber optic": "fiber optic internet"}.get(
        facts["internet_service"], "phone")
    offer_txt = " and ".join(facts["offers"])
    return (
        f"Hello, this is [Agent name] from [Company]. Thank you for {_time_with_us(facts['tenure_months'])}. "
        f"I'm calling to make sure your {service} service is working well for you at your current bill of "
        f"USD {facts['monthly_charges']:.2f} a month. "
        f"As a valued customer, we can offer {offer_txt}. "
        f"There is no obligation, and I can explain exactly how it would look on your next bill. "
        f"Would you like me to set that up for you today?"
    )


def llm_script(facts: dict) -> str | None:
    """Call the Anthropic API when ANTHROPIC_API_KEY is set; return None on any failure."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(model=MODEL, max_tokens=400, temperature=0.4,
                                     messages=[{"role": "user", "content": build_prompt(facts)}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    except Exception as exc:  # network / quota / model errors must never break the dashboard
        print(f"[genai] LLM call failed, falling back to template: {exc}")
        return None


# ----------------------------------------------------------------------------------------
# 3) Verification layer
# ----------------------------------------------------------------------------------------
@dataclass
class Verification:
    passed: bool
    issues: list[str] = field(default_factory=list)


def verify_script(text: str, facts: dict) -> Verification:
    issues: list[str] = []
    low = text.lower()

    # (a) length
    n_words = len(text.split())
    if not 60 <= n_words <= 150:
        issues.append(f"Length {n_words} words is outside the 60-150 word range")

    # (b) forbidden / sensitive terms
    for term in FORBIDDEN_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", low):
            issues.append(f"Contains forbidden term: '{term}'")

    # (c) every currency amount must be the customer's real bill or an approved credit
    allowed_money = {round(facts["monthly_charges"], 2), round(facts["monthly_charges"]), 5.0}
    for m in re.findall(r"(?:usd|us\$|\$)\s?(\d+(?:\.\d+)?)", low):
        if float(m) not in allowed_money:
            issues.append(f"Unapproved amount: USD {m}")

    # (d) every percentage must come from an approved offer
    approved_text = " ".join(facts["offers"])
    allowed_pct = set(re.findall(r"(\d+)\s?%", approved_text))
    for pct in re.findall(r"(\d+)\s?%", low):
        if pct not in allowed_pct:
            issues.append(f"Unapproved percentage: {pct}%")

    # (e) durations must match an approved offer or the customer's real tenure
    allowed_months = set(re.findall(r"(\d+)[- ]months?", approved_text)) | {str(facts["tenure_months"])}
    for mth in re.findall(r"(\d+)[- ]months?", low):
        if mth not in allowed_months:
            issues.append(f"Unapproved duration: {mth} months")
    allowed_years = {str(facts["tenure_months"] // 12)} | set(re.findall(r"(\d+)[- ]year", approved_text))
    for yr in re.findall(r"(\d+)[- ]years?", low):
        if yr not in allowed_years:
            issues.append(f"Unapproved duration: {yr} years")

    # (f) must actually mention at least one of the approved offers (keyword check)
    keywords = {"10% discount": "discount", "paper-bill": "automatic payment", "Tech Support": "tech support",
                "Online Security": "online security", "plan review": "plan review"}
    wanted = [kw for key, kw in keywords.items() if any(key in o for o in facts["offers"])]
    if wanted and not any(kw in low for kw in wanted):
        issues.append("Does not mention any of the approved offers")

    # (g) must end with a question (clear call to action)
    if not text.strip().endswith("?"):
        issues.append("Does not end with a clear question / call to action")

    return Verification(passed=not issues, issues=issues)


# ----------------------------------------------------------------------------------------
# 4) Public entry point
# ----------------------------------------------------------------------------------------
def generate_retention_script(row) -> dict:
    facts = customer_facts(row)
    text = llm_script(facts)
    source = "LLM"
    if text is None:
        text, source = template_script(facts), "Template (offline)"
    ver = verify_script(text, facts)
    if not ver.passed and source == "LLM":          # one safe retry path: replace bad LLM output
        text, source = template_script(facts), "Template (LLM output rejected)"
        ver = verify_script(text, facts)
    return {"script": text, "source": source, "verified": ver.passed, "issues": ver.issues, "facts": facts}
