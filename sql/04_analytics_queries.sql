-- sql/04_analytics_queries.sql
-- Reusable parameterised analytical queries.
-- In application code, use SQLAlchemy text() with bound params.

-- ── Q1: Average speed by sensor and hour ─────────────────────────
-- :start_date TIMESTAMPTZ, :end_date TIMESTAMPTZ
SELECT
    sensor_id,
    EXTRACT(HOUR FROM recorded_at) AS hour_of_day,
    ROUND(AVG(observed_speed)::NUMERIC, 2)  AS avg_speed_mph,
    COUNT(*)                                 AS reading_count
FROM traffic_readings
WHERE recorded_at BETWEEN :start_date AND :end_date
GROUP BY sensor_id, hour_of_day
ORDER BY sensor_id, hour_of_day;

-- ── Q2: Lowest-speed sensors (overall) ───────────────────────────
-- :top_n INTEGER
SELECT
    sensor_id,
    ROUND(AVG(observed_speed)::NUMERIC, 2) AS avg_speed_mph,
    ROUND(MIN(observed_speed)::NUMERIC, 2) AS min_speed_mph,
    COUNT(*) AS observations
FROM traffic_readings
GROUP BY sensor_id
ORDER BY avg_speed_mph ASC
LIMIT :top_n;

-- ── Q3: Peak vs off-peak comparison ──────────────────────────────
-- Peak = 07:00–09:00 and 17:00–19:00
SELECT
    CASE
        WHEN EXTRACT(HOUR FROM recorded_at) IN (7,8,17,18) THEN 'Peak'
        ELSE 'Off-Peak'
    END AS period,
    ROUND(AVG(observed_speed)::NUMERIC, 2) AS avg_speed_mph,
    COUNT(*) AS observations
FROM traffic_readings
GROUP BY period
ORDER BY avg_speed_mph;

-- ── Q4: Daily traffic pattern (weekday vs weekend) ────────────────
SELECT
    TO_CHAR(recorded_at, 'Dy')            AS day_name,
    EXTRACT(DOW FROM recorded_at)         AS day_of_week,
    CASE WHEN EXTRACT(DOW FROM recorded_at) IN (0,6) THEN 'Weekend'
         ELSE 'Weekday' END               AS day_type,
    ROUND(AVG(observed_speed)::NUMERIC, 2) AS avg_speed_mph,
    COUNT(*) AS observations
FROM traffic_readings
GROUP BY day_name, day_of_week, day_type
ORDER BY day_of_week;

-- ── Q5: Sensor-level statistics ───────────────────────────────────
-- :sensor_id VARCHAR
SELECT
    sensor_id,
    ROUND(AVG(observed_speed)::NUMERIC, 2)    AS avg_speed_mph,
    ROUND(MIN(observed_speed)::NUMERIC, 2)    AS min_speed_mph,
    ROUND(MAX(observed_speed)::NUMERIC, 2)    AS max_speed_mph,
    ROUND(STDDEV(observed_speed)::NUMERIC, 2) AS std_speed_mph,
    COUNT(*)                                   AS total_readings,
    SUM(CASE WHEN outlier_flag = 1 THEN 1 ELSE 0 END) AS outlier_count
FROM traffic_readings
WHERE sensor_id = :sensor_id
GROUP BY sensor_id;

-- ── Q6: Predicted vs actual speed for a sensor ───────────────────
-- :sensor_id VARCHAR, :horizon_min INTEGER
SELECT
    tr.recorded_at,
    tr.observed_speed         AS actual_speed_mph,
    cp.predicted_speed_mph,
    cp.congestion_level,
    ABS(tr.observed_speed - cp.predicted_speed_mph) AS abs_error
FROM traffic_readings tr
JOIN congestion_predictions cp
  ON tr.sensor_id = cp.sensor_id
  AND cp.forecast_for BETWEEN tr.recorded_at - INTERVAL '3 minutes'
                          AND tr.recorded_at + INTERVAL '3 minutes'
WHERE tr.sensor_id       = :sensor_id
  AND cp.forecast_horizon_min = :horizon_min
ORDER BY tr.recorded_at DESC
LIMIT 200;

-- ── Q7: Congestion-level summary ─────────────────────────────────
SELECT
    congestion_level,
    COUNT(*)                                         AS prediction_count,
    ROUND(AVG(predicted_speed_mph)::NUMERIC, 2)     AS avg_predicted_speed,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM congestion_predictions
GROUP BY congestion_level
ORDER BY avg_predicted_speed DESC;

-- ── Q8: Model evaluation results ─────────────────────────────────
SELECT
    model_name,
    model_version,
    horizon_min,
    test_mae,
    test_rmse,
    n_sensors_trained,
    trained_at
FROM model_registry
ORDER BY horizon_min, test_mae;
