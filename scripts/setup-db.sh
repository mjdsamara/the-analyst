#!/bin/bash
# PostgreSQL setup script for The Analyst

set -e

echo "=== The Analyst - Database Setup ==="
echo ""

# Configuration
DB_NAME="${DB_NAME:-analyst}"
DB_USER="${DB_USER:-analyst}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

echo "Setting up database: $DB_NAME"
echo "Host: $DB_HOST:$DB_PORT"
echo ""

# Check if PostgreSQL is running
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" > /dev/null 2>&1; then
    echo "PostgreSQL is not running at $DB_HOST:$DB_PORT"
    echo "Please start PostgreSQL and try again."
    exit 1
fi

# Create database if it doesn't exist
if psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    echo "Database '$DB_NAME' already exists."
else
    echo "Creating database '$DB_NAME'..."
    createdb -h "$DB_HOST" -p "$DB_PORT" -U postgres "$DB_NAME"
fi

# Create user if it doesn't exist
psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1 || \
    psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "CREATE USER $DB_USER WITH PASSWORD 'analyst_password';"

# Grant privileges
psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

# Create tables
echo "Creating tables..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" << 'EOF'

-- Analysis history table
CREATE TABLE IF NOT EXISTS analysis_history (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    parameters JSONB,
    results JSONB,
    metadata JSONB
);

-- Data checksums table
CREATE TABLE IF NOT EXISTS data_checksums (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL UNIQUE,
    checksum VARCHAR(64) NOT NULL,
    file_size BIGINT,
    first_seen TIMESTAMP NOT NULL DEFAULT NOW(),
    last_verified TIMESTAMP NOT NULL DEFAULT NOW()
);

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL UNIQUE,
    preferences JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Workflow states table
CREATE TABLE IF NOT EXISTS workflow_states (
    id SERIAL PRIMARY KEY,
    workflow_id UUID NOT NULL UNIQUE,
    session_id UUID NOT NULL,
    phase VARCHAR(50) NOT NULL,
    state JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_analysis_session ON analysis_history(session_id);
CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_history(analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_status ON analysis_history(status);
CREATE INDEX IF NOT EXISTS idx_workflow_session ON workflow_states(session_id);

EOF

echo ""
echo "Database setup complete!"
echo ""
echo "Connection string:"
echo "postgresql://$DB_USER:analyst_password@$DB_HOST:$DB_PORT/$DB_NAME"
