-- sql/01_create_database.sql
-- Run as superuser BEFORE running subsequent scripts.
-- Example: psql -U postgres -f sql/01_create_database.sql

-- Create database
CREATE DATABASE smart_city_traffic
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0
    CONNECTION LIMIT = -1;

-- Connect to the new database and enable PostGIS (optional)
\connect smart_city_traffic;
-- CREATE EXTENSION IF NOT EXISTS postgis;  -- Uncomment if PostGIS is installed
