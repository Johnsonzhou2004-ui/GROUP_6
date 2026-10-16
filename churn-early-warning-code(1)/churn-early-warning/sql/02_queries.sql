-- =====================================================================
-- 02_queries.sql  |  Named analytical queries used by src/analysis.py
-- Each block starts with "-- name: <query_name>" so Python can load it.
-- Techniques shown: JOIN, WHERE, GROUP BY, HAVING, CASE, window functions, CTE, subquery.
-- =====================================================================

-- name: portfolio_kpis
-- Headline KPIs for the whole customer base.
SELECT
    COUNT(*)                                         AS customers,
    SUM(churn)                                       AS churned,
    ROUND(100.0 * AVG(churn), 2)                     AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                   AS avg_monthly_charge,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges END), 0) AS monthly_revenue_lost
FROM v_customer_360;

-- name: churn_by_contract
-- Which contract type carries the most cancellation risk?
SELECT
    contract,
    COUNT(*)                                         AS customers,
    SUM(churn)                                       AS churned,
    ROUND(100.0 * AVG(churn), 2)                     AS churn_rate_pct,
    ROUND(100.0 * SUM(churn) / (SELECT SUM(churn) FROM customers), 1) AS share_of_all_churn_pct
FROM v_customer_360
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- name: churn_by_tenure_band
SELECT
    tenure_band,
    COUNT(*)                                         AS customers,
    ROUND(100.0 * AVG(churn), 2)                     AS churn_rate_pct
FROM v_customer_360
GROUP BY tenure_band
ORDER BY MIN(tenure_months);

-- name: churn_by_internet_service
SELECT
    internet_service,
    COUNT(*)                                         AS customers,
    ROUND(100.0 * AVG(churn), 2)                     AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                   AS avg_monthly_charge
FROM v_customer_360
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;

-- name: churn_by_payment_method
SELECT
    payment_method,
    COUNT(*)                                         AS customers,
    ROUND(100.0 * AVG(churn), 2)                     AS churn_rate_pct
FROM v_customer_360
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;

-- name: churn_by_support_addons
-- Do protective add-ons (security / tech support) reduce churn? Internet customers only.
SELECT
    CASE WHEN online_security = 'Yes' THEN 'Yes' ELSE 'No' END AS has_online_security,
    CASE WHEN tech_support    = 'Yes' THEN 'Yes' ELSE 'No' END AS has_tech_support,
    COUNT(*)                                         AS customers,
    ROUND(100.0 * AVG(churn), 2)                     AS churn_rate_pct
FROM v_customer_360
WHERE internet_service <> 'No'
GROUP BY has_online_security, has_tech_support
ORDER BY churn_rate_pct DESC;

-- name: high_risk_segments
-- Segments (contract x internet x tenure band) with at least 100 customers,
-- ranked by churn rate. HAVING removes small, noisy segments.
SELECT
    contract, internet_service, tenure_band,
    COUNT(*)                                         AS customers,
    ROUND(100.0 * AVG(churn), 1)                     AS churn_rate_pct,
    ROUND(SUM(monthly_charges), 0)                   AS monthly_revenue_at_stake
FROM v_customer_360
GROUP BY contract, internet_service, tenure_band
HAVING COUNT(*) >= 100
ORDER BY churn_rate_pct DESC
LIMIT 10;

-- name: churn_vs_portfolio_average
-- CTE + window function: how far is each contract/internet segment from the portfolio average?
WITH seg AS (
    SELECT contract, internet_service,
           COUNT(*) AS customers,
           AVG(churn) AS churn_rate
    FROM v_customer_360
    GROUP BY contract, internet_service
)
SELECT
    contract, internet_service, customers,
    ROUND(100.0 * churn_rate, 1) AS churn_rate_pct,
    ROUND(100.0 * (churn_rate - (SELECT AVG(churn) FROM customers)), 1) AS pts_vs_average,
    RANK() OVER (ORDER BY churn_rate DESC) AS risk_rank
FROM seg
ORDER BY risk_rank;

-- name: risk_tier_summary
-- Uses the Python model output (churn_scores) joined back to billing.
SELECT
    risk_tier,
    COUNT(*)                                         AS customers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM churn_scores), 1) AS pct_of_base,
    ROUND(AVG(churn_prob), 3)                        AS avg_predicted_prob,
    ROUND(100.0 * AVG(churn), 1)                     AS actual_churn_rate_pct,
    ROUND(SUM(monthly_charges), 0)                   AS monthly_revenue
FROM v_customer_360
WHERE risk_tier IS NOT NULL
GROUP BY risk_tier
ORDER BY CASE risk_tier WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 ELSE 4 END;

-- name: top_at_risk_customers
-- The call list: customers who are STILL ACTIVE (churn = 0) and rated High/Critical,
-- ranked by expected monthly loss = churn probability x monthly charge.
SELECT
    customer_id, contract, internet_service, tenure_months, monthly_charges,
    ROUND(churn_prob, 3)                             AS churn_prob,
    risk_tier,
    ROUND(churn_prob * monthly_charges, 2)           AS expected_monthly_loss,
    top_drivers
FROM v_customer_360
WHERE churn = 0
  AND risk_tier IN ('High', 'Critical')
ORDER BY expected_monthly_loss DESC
LIMIT 25;

-- name: active_base_by_tier
-- Operational view: the live customer base (still active) by risk tier, with revenue exposed.
SELECT
    risk_tier,
    COUNT(*)                                         AS active_customers,
    ROUND(SUM(monthly_charges), 0)                   AS monthly_revenue,
    ROUND(SUM(churn_prob * monthly_charges), 0)      AS expected_monthly_loss
FROM v_customer_360
WHERE churn = 0
GROUP BY risk_tier
ORDER BY CASE risk_tier WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 ELSE 4 END;
