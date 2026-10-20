-- sql/02_create_tables.sql
-- Run against the smart_city_traffic database.
-- Example: psql -U postgres -d smart_city_traffic -f sql/02_create_tables.sql

-- ── Sensors (replaces road_segments when GPS coords unavailable) ──
CREATE TABLE IF NOT EXISTS sensors (
    sensor_id         VARCHAR(20)  PRIMARY KEY,
    area_name         VARCHAR(100),
    road_type         VARCHAR(50),
    latitude          NUMERIC(9,6),   -- NULL when coordinates are unverified
    longitude         NUMERIC(9,6),   -- NULL when coordinates are unverified
    speed_limit_mph   NUMERIC(5,1),   -- NULL when unverified
    notes             TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE sensors IS
    'PEMS-BAY sensor registry. GPS coordinates and speed limits are optional '
    'and must only be populated from verified external sources.';

-- ── Traffic Readings ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS traffic_readings (
    reading_id        BIGSERIAL    PRIMARY KEY,
    sensor_id         VARCHAR(20)  NOT NULL REFERENCES sensors(sensor_id),
    recorded_at       TIMESTAMPTZ  NOT NULL,
    observed_speed    NUMERIC(6,2) NOT NULL,  -- mph
    vehicle_count     INTEGER,                -- nullable (not in PEMS-BAY CSV)
    occupancy_pct     NUMERIC(5,2),           -- nullable
    rainfall_mm       NUMERIC(6,2),           -- nullable external data
    temperature_c     NUMERIC(5,2),           -- nullable external data
    incident_flag     BOOLEAN      DEFAULT FALSE,
    outlier_flag      SMALLINT     DEFAULT 0,
    created_at        TIMESTAMPTZ  DEFAULT NOW()
);
COMMENT ON TABLE traffic_readings IS
    'Speed observations from PEMS-BAY. vehicle_count, occupancy_pct, rainfall_mm, '
    'and temperature_c are NULL when not available in the source dataset.';

-- ── Congestion Predictions ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS congestion_predictions (
    prediction_id       BIGSERIAL   PRIMARY KEY,
    sensor_id           VARCHAR(20) NOT NULL REFERENCES sensors(sensor_id),
    generated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    forecast_for        TIMESTAMPTZ NOT NULL,
    forecast_horizon_min INTEGER    NOT NULL CHECK (forecast_horizon_min IN (15, 30, 60)),
    predicted_speed_mph NUMERIC(6,2) NOT NULL,
    congestion_level    VARCHAR(20)  NOT NULL,  -- free_flow | moderate | heavy | severe
    model_name          VARCHAR(50)  NOT NULL,
    model_version       VARCHAR(30)  NOT NULL
);

-- ── Model Registry ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    model_id          SERIAL       PRIMARY KEY,
    model_name        VARCHAR(50)  NOT NULL,
    model_version     VARCHAR(30)  NOT NULL UNIQUE,
    horizon_min       INTEGER      NOT NULL,
    test_mae          NUMERIC(8,4),
    test_rmse         NUMERIC(8,4),
    n_sensors_trained INTEGER,
    trained_at        TIMESTAMPTZ  NOT NULL,
    model_path        TEXT,
    scaler_path       TEXT,
    feature_cols      TEXT         -- JSON array stored as text
);
