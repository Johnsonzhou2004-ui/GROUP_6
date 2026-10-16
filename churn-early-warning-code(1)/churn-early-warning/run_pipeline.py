"""One-command pipeline:  CSV -> SQLite -> model scores -> SQL analysis + figures.

Usage:  python run_pipeline.py
Then:   streamlit run app/dashboard.py
"""
from src import analysis, etl, model


def main() -> None:
    print("== 1/3  ETL: load CSV into SQLite ==")
    etl.main()
    print("== 2/3  Model: train, evaluate, score ==")
    model.main()
    print("== 3/3  Analysis: SQL queries and figures ==")
    analysis.main()
    print("\nDone. Launch the dashboard with:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
