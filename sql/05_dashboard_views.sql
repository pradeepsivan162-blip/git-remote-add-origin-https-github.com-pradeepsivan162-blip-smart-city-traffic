-- sql/05_dashboard_views.sql
-- Materialised views for dashboard KPI tiles.
-- Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sensor_summary;

-- ── Sensor summary (refreshed daily) ─────────────────────────────
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_sensor_summary AS
SELECT
    sensor_id,
    ROUND(AVG(observed_speed)::NUMERIC, 2)    AS avg_speed_mph,
    ROUND(MIN(observed_speed)::NUMERIC, 2)    AS min_speed_mph,
    ROUND(MAX(observed_speed)::NUMERIC, 2)    AS max_speed_mph,
    ROUND(STDDEV(observed_speed)::NUMERIC, 2) AS std_speed_mph,
    COUNT(*)                                   AS n_readings,
    MIN(recorded_at)                           AS first_reading,
    MAX(recorded_at)                           AS last_reading
FROM traffic_readings
GROUP BY sensor_id
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS uidx_mv_sensor_summary
    ON mv_sensor_summary (sensor_id);

-- ── Hourly speed heatmap view ─────────────────────────────────────
CREATE OR REPLACE VIEW v_hourly_speed AS
SELECT
    sensor_id,
    EXTRACT(HOUR FROM recorded_at)             AS hour_of_day,
    EXTRACT(DOW  FROM recorded_at)             AS day_of_week,
    ROUND(AVG(observed_speed)::NUMERIC, 2)     AS avg_speed_mph
FROM traffic_readings
GROUP BY sensor_id, hour_of_day, day_of_week;

-- ── Latest predictions per sensor ────────────────────────────────
CREATE OR REPLACE VIEW v_latest_predictions AS
SELECT DISTINCT ON (sensor_id, forecast_horizon_min)
    sensor_id,
    forecast_horizon_min,
    predicted_speed_mph,
    congestion_level,
    forecast_for,
    generated_at,
    model_name
FROM congestion_predictions
ORDER BY sensor_id, forecast_horizon_min, generated_at DESC;
