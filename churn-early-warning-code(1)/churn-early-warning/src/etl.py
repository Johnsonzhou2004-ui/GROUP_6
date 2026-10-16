"""ETL: raw Telco CSV  ->  cleaned, validated rows  ->  SQLite database.

Run:  python -m src.etl
"""
import sqlite3

import pandas as pd

from src.config import DB_PATH, RAW_CSV, SQL_DIR


def load_and_clean(csv_path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # --- data-quality fixes -------------------------------------------------
    # TotalCharges is stored as text; 11 rows are blank. All of them have
    # tenure == 0 (new customers who have not been billed yet), so NULL is the
    # correct value - not 0 and not an imputed mean.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    blank = df["TotalCharges"].isna()
    assert (df.loc[blank, "tenure"] == 0).all(), "Unexpected blank TotalCharges with tenure > 0"

    # Label as 0/1
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # --- integrity checks ---------------------------------------------------
    assert df["customerID"].is_unique, "Duplicate customerID"
    assert df["Churn"].notna().all(), "Unmapped churn label"
    assert (df["MonthlyCharges"] >= 0).all()
    return df


def build_database(df: pd.DataFrame, db_path=DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript((SQL_DIR / "01_schema.sql").read_text())

    customers = df[["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "Churn"]]
    services = df[["customerID", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                   "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]]
    billing = df[["customerID", "Contract", "PaperlessBilling", "PaymentMethod",
                  "MonthlyCharges", "TotalCharges"]]

    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?)", customers.itertuples(index=False, name=None))
    con.executemany("INSERT INTO services VALUES (?,?,?,?,?,?,?,?,?,?)", services.itertuples(index=False, name=None))
    # NaN -> None so SQLite stores NULL
    rows = [tuple(None if pd.isna(v) else v for v in r) for r in billing.itertuples(index=False, name=None)]
    con.executemany("INSERT INTO billing VALUES (?,?,?,?,?,?)", rows)
    con.commit()

    # reconciliation: every table must hold exactly as many rows as the CSV
    for t in ("customers", "services", "billing"):
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        assert n == len(df), f"Row count mismatch in {t}: {n} vs {len(df)}"
    con.close()


def main() -> None:
    df = load_and_clean()
    build_database(df)
    print(f"[etl] loaded {len(df):,} customers into {DB_PATH}")


if __name__ == "__main__":
    main()
