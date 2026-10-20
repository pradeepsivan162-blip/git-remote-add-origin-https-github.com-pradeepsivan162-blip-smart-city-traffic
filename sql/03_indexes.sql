-- sql/03_indexes.sql
-- Performance indexes for dashboard queries.

-- traffic_readings — most common lookup patterns
CREATE INDEX IF NOT EXISTS idx_tr_sensor_time
    ON traffic_readings (sensor_id, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_tr_recorded_at
    ON traffic_readings (recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_tr_speed
    ON traffic_readings (observed_speed);

-- congestion_predictions
CREATE INDEX IF NOT EXISTS idx_cp_sensor_forecast
    ON congestion_predictions (sensor_id, forecast_for DESC);

CREATE INDEX IF NOT EXISTS idx_cp_generated
    ON congestion_predictions (generated_at DESC);

CREATE INDEX IF NOT EXISTS idx_cp_horizon
    ON congestion_predictions (forecast_horizon_min);

-- Partial index: only flag anomalies
CREATE INDEX IF NOT EXISTS idx_tr_outliers
    ON traffic_readings (sensor_id, recorded_at)
    WHERE outlier_flag = 1;
