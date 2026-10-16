-- =====================================================================
-- 01_schema.sql  |  Customer Churn Early Warning  |  SQLite schema
-- Normalised star-style layout: one customer dimension, one services
-- table, one billing/contract fact table, and one model-score table.
-- =====================================================================

PRAGMA foreign_keys = ON;

DROP VIEW  IF EXISTS v_customer_360;
DROP TABLE IF EXISTS churn_scores;
DROP TABLE IF EXISTS billing;
DROP TABLE IF EXISTS services;
DROP TABLE IF EXISTS customers;

-- Who the customer is (demographics + outcome label)
CREATE TABLE customers (
    customer_id     TEXT PRIMARY KEY,
    gender          TEXT NOT NULL CHECK (gender IN ('Male', 'Female')),
    senior_citizen  INTEGER NOT NULL CHECK (senior_citizen IN (0, 1)),
    partner         TEXT NOT NULL CHECK (partner IN ('Yes', 'No')),
    dependents      TEXT NOT NULL CHECK (dependents IN ('Yes', 'No')),
    tenure_months   INTEGER NOT NULL CHECK (tenure_months >= 0),
    churn           INTEGER NOT NULL CHECK (churn IN (0, 1))  -- 1 = churned
);

-- What the customer subscribes to
CREATE TABLE services (
    customer_id       TEXT PRIMARY KEY REFERENCES customers (customer_id),
    phone_service     TEXT NOT NULL,
    multiple_lines    TEXT NOT NULL,
    internet_service  TEXT NOT NULL,
    online_security   TEXT NOT NULL,
    online_backup     TEXT NOT NULL,
    device_protection TEXT NOT NULL,
    tech_support      TEXT NOT NULL,
    streaming_tv      TEXT NOT NULL,
    streaming_movies  TEXT NOT NULL
);

-- How the customer pays
CREATE TABLE billing (
    customer_id       TEXT PRIMARY KEY REFERENCES customers (customer_id),
    contract          TEXT NOT NULL CHECK (contract IN ('Month-to-month', 'One year', 'Two year')),
    paperless_billing TEXT NOT NULL CHECK (paperless_billing IN ('Yes', 'No')),
    payment_method    TEXT NOT NULL,
    monthly_charges   REAL NOT NULL CHECK (monthly_charges >= 0),
    total_charges     REAL            -- NULL for brand-new customers (tenure = 0)
);

-- Output of the Python model (refreshed each pipeline run)
CREATE TABLE churn_scores (
    customer_id  TEXT PRIMARY KEY REFERENCES customers (customer_id),
    churn_prob   REAL NOT NULL CHECK (churn_prob BETWEEN 0 AND 1),
    risk_tier    TEXT NOT NULL CHECK (risk_tier IN ('Low', 'Medium', 'High', 'Critical')),
    top_drivers  TEXT,              -- pipe-separated plain-language reasons
    scored_at    TEXT NOT NULL
);

CREATE INDEX idx_billing_contract   ON billing (contract);
CREATE INDEX idx_services_internet  ON services (internet_service);
CREATE INDEX idx_scores_tier        ON churn_scores (risk_tier);

-- One flat, analysis-ready view joining everything
CREATE VIEW v_customer_360 AS
SELECT
    c.customer_id,
    c.gender, c.senior_citizen, c.partner, c.dependents,
    c.tenure_months,
    CASE
        WHEN c.tenure_months <= 12 THEN '0-12 months'
        WHEN c.tenure_months <= 24 THEN '13-24 months'
        WHEN c.tenure_months <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_band,
    s.phone_service, s.multiple_lines, s.internet_service,
    s.online_security, s.online_backup, s.device_protection,
    s.tech_support, s.streaming_tv, s.streaming_movies,
    b.contract, b.paperless_billing, b.payment_method,
    b.monthly_charges, b.total_charges,
    c.churn,
    sc.churn_prob, sc.risk_tier, sc.top_drivers
FROM customers c
JOIN services s  ON s.customer_id = c.customer_id
JOIN billing  b  ON b.customer_id = c.customer_id
LEFT JOIN churn_scores sc ON sc.customer_id = c.customer_id;
