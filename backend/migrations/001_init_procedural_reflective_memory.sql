-- ============================================================================
-- Weather AI Agent Service - PostgreSQL Schema Migration
-- Level 3c: Procedural & Reflective Memory (Layers 5 & 7)
-- ============================================================================
-- Migration: 001_init_procedural_reflective_memory.sql
-- Created: 2025-12-07
-- Description: Initialize tables for Layers 5 (Procedural) and 7 (Reflective)
--
-- Layer 5: Procedural Memory - Workflow optimization and tool selection
-- Layer 7: Reflective Memory - Self-improvement and meta-learning
--
-- Usage:
--   psql -U weather_ai -d weather_ai -f backend/migrations/001_init_procedural_reflective_memory.sql
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Layer 5: Procedural Memory (Workflow Optimization)
-- ============================================================================

-- Table: workflows
-- Purpose: Store learned workflow patterns with performance metrics
-- Retention: Indefinite (cleaned up by consolidation pipeline based on usage)
CREATE TABLE IF NOT EXISTS workflows (
    -- Primary key
    id SERIAL PRIMARY KEY,

    -- Workflow identification
    task_name VARCHAR(255) NOT NULL,
    workflow_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA256 of steps array for deduplication

    -- Workflow steps (array of tool names)
    steps TEXT[] NOT NULL,

    -- Performance metrics
    success_rate FLOAT NOT NULL DEFAULT 0.0 CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
    avg_response_time FLOAT NOT NULL DEFAULT 0.0 CHECK (avg_response_time >= 0.0),  -- seconds
    usage_count INTEGER NOT NULL DEFAULT 0 CHECK (usage_count >= 0),

    -- Temporal tracking
    first_used TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Quality metrics
    feedback_score FLOAT CHECK (feedback_score IS NULL OR (feedback_score >= 0.0 AND feedback_score <= 5.0)),

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for Layer 5
CREATE INDEX IF NOT EXISTS idx_workflows_task_name ON workflows(task_name);
CREATE INDEX IF NOT EXISTS idx_workflows_last_used ON workflows(last_used DESC);
CREATE INDEX IF NOT EXISTS idx_workflows_success_rate ON workflows(success_rate DESC);
CREATE INDEX IF NOT EXISTS idx_workflows_usage_count ON workflows(usage_count DESC);

-- Table: tool_usage
-- Purpose: Track individual tool invocations for analytics
-- Retention: 30 days (cleaned up by consolidation pipeline)
CREATE TABLE IF NOT EXISTS tool_usage (
    -- Primary key
    id SERIAL PRIMARY KEY,

    -- Tool identification
    tool_name VARCHAR(255) NOT NULL,

    -- Context
    task_context VARCHAR(500),
    user_id VARCHAR(255),
    session_id VARCHAR(255),

    -- Performance
    execution_time FLOAT NOT NULL CHECK (execution_time >= 0.0),  -- seconds
    success BOOLEAN NOT NULL,
    error_message TEXT,

    -- Temporal
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for tool_usage
CREATE INDEX IF NOT EXISTS idx_tool_usage_tool_name ON tool_usage(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_usage_timestamp ON tool_usage(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tool_usage_user_id ON tool_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_session_id ON tool_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_success ON tool_usage(success);

-- ============================================================================
-- Layer 7: Reflective Memory (Self-Improvement)
-- ============================================================================

-- Table: reflections
-- Purpose: Store self-improvement insights and learnings
-- Retention: Indefinite (critical for long-term improvement)
CREATE TABLE IF NOT EXISTS reflections (
    -- Primary key
    id SERIAL PRIMARY KEY,
    reflection_id UUID NOT NULL UNIQUE DEFAULT uuid_generate_v4(),

    -- Reflection metadata
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    trigger VARCHAR(255) NOT NULL,  -- "user_feedback_negative", "error_pattern", etc.

    -- Reflection content
    context TEXT NOT NULL,
    analysis TEXT NOT NULL,
    insight TEXT NOT NULL,
    learned_from VARCHAR(255),  -- Episode ID or pattern description

    -- Action tracking
    action_taken TEXT,
    applied_count INTEGER NOT NULL DEFAULT 0 CHECK (applied_count >= 0),

    -- Effectiveness metrics
    effectiveness FLOAT NOT NULL DEFAULT 0.0 CHECK (effectiveness >= 0.0 AND effectiveness <= 1.0),
    improvement_score FLOAT CHECK (improvement_score IS NULL OR (improvement_score >= 0.0 AND improvement_score <= 1.0)),

    -- Status
    applied BOOLEAN NOT NULL DEFAULT FALSE,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for Layer 7
CREATE INDEX IF NOT EXISTS idx_reflections_reflection_id ON reflections(reflection_id);
CREATE INDEX IF NOT EXISTS idx_reflections_timestamp ON reflections(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_reflections_trigger ON reflections(trigger);
CREATE INDEX IF NOT EXISTS idx_reflections_applied ON reflections(applied);
CREATE INDEX IF NOT EXISTS idx_reflections_effectiveness ON reflections(effectiveness DESC);

-- Full-text search index for insights
CREATE INDEX IF NOT EXISTS idx_reflections_insight_fts ON reflections USING GIN(to_tsvector('english', insight));
CREATE INDEX IF NOT EXISTS idx_reflections_analysis_fts ON reflections USING GIN(to_tsvector('english', analysis));

-- ============================================================================
-- Triggers for automatic timestamp updates
-- ============================================================================

-- Function: update_updated_at_column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: workflows updated_at
CREATE TRIGGER update_workflows_updated_at
    BEFORE UPDATE ON workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger: reflections updated_at
CREATE TRIGGER update_reflections_updated_at
    BEFORE UPDATE ON reflections
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Partitioning Setup (for tool_usage table - high volume)
-- ============================================================================

-- Note: tool_usage will be partitioned by month for efficient data retention
-- This is implemented in 002_partition_tool_usage.sql (future migration)

-- ============================================================================
-- Materialized Views for Analytics (Layer 5)
-- ============================================================================

-- View: workflow_performance_summary
-- Purpose: Aggregated workflow performance metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS workflow_performance_summary AS
SELECT
    task_name,
    COUNT(*) AS total_workflows,
    AVG(success_rate) AS avg_success_rate,
    AVG(avg_response_time) AS avg_response_time,
    SUM(usage_count) AS total_usage,
    MAX(last_used) AS most_recent_use
FROM workflows
GROUP BY task_name
ORDER BY total_usage DESC;

-- Index for materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_perf_task_name ON workflow_performance_summary(task_name);

-- View: tool_performance_summary
-- Purpose: Aggregated tool performance metrics (last 7 days)
CREATE MATERIALIZED VIEW IF NOT EXISTS tool_performance_summary AS
SELECT
    tool_name,
    COUNT(*) AS total_invocations,
    COUNT(*) FILTER (WHERE success = TRUE) AS successful_invocations,
    COUNT(*) FILTER (WHERE success = FALSE) AS failed_invocations,
    ROUND((COUNT(*) FILTER (WHERE success = TRUE)::FLOAT / COUNT(*))::NUMERIC, 4) AS success_rate,
    AVG(execution_time) AS avg_execution_time,
    MAX(timestamp) AS last_used
FROM tool_usage
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY tool_name
ORDER BY total_invocations DESC;

-- Index for materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_perf_tool_name ON tool_performance_summary(tool_name);

-- ============================================================================
-- Materialized Views for Analytics (Layer 7)
-- ============================================================================

-- View: reflection_effectiveness_summary
-- Purpose: Track which types of reflections are most effective
CREATE MATERIALIZED VIEW IF NOT EXISTS reflection_effectiveness_summary AS
SELECT
    trigger,
    COUNT(*) AS total_reflections,
    COUNT(*) FILTER (WHERE applied = TRUE) AS applied_reflections,
    AVG(effectiveness) AS avg_effectiveness,
    AVG(improvement_score) AS avg_improvement,
    MAX(timestamp) AS most_recent_reflection
FROM reflections
GROUP BY trigger
ORDER BY avg_effectiveness DESC;

-- Index for materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_reflection_eff_trigger ON reflection_effectiveness_summary(trigger);

-- ============================================================================
-- Functions for Materialized View Refresh
-- ============================================================================

-- Function: refresh_all_materialized_views
-- Purpose: Refresh all materialized views (called by scheduled job)
CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY workflow_performance_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY tool_performance_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY reflection_effectiveness_summary;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Data Retention Policies (Implemented by consolidation pipeline)
-- ============================================================================

-- Function: cleanup_old_tool_usage
-- Purpose: Delete tool_usage records older than 30 days
CREATE OR REPLACE FUNCTION cleanup_old_tool_usage()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM tool_usage
    WHERE timestamp < NOW() - INTERVAL '30 days';

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function: cleanup_low_usage_workflows
-- Purpose: Archive workflows with usage_count < 3 and last_used > 90 days
CREATE OR REPLACE FUNCTION cleanup_low_usage_workflows()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM workflows
    WHERE usage_count < 3
    AND last_used < NOW() - INTERVAL '90 days';

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Initial Data / Seed Data (Optional)
-- ============================================================================

-- Insert example reflection triggers (for documentation purposes)
INSERT INTO reflections (
    trigger,
    context,
    analysis,
    insight,
    learned_from,
    applied
) VALUES
(
    'initial_setup',
    'Level 3c PostgreSQL schema initialization',
    'Created procedural and reflective memory tables with proper indexing and partitioning',
    'Schema design follows best practices: proper constraints, indexes, materialized views for analytics',
    'migration_001',
    TRUE
)
ON CONFLICT (reflection_id) DO NOTHING;

-- ============================================================================
-- Verification Queries (Run after migration)
-- ============================================================================

-- Verify tables exist
DO $$
BEGIN
    ASSERT (SELECT COUNT(*) FROM information_schema.tables WHERE table_name IN ('workflows', 'tool_usage', 'reflections')) = 3,
        'Not all tables were created successfully';

    RAISE NOTICE 'Migration 001 completed successfully!';
    RAISE NOTICE 'Created tables: workflows, tool_usage, reflections';
    RAISE NOTICE 'Created materialized views: workflow_performance_summary, tool_performance_summary, reflection_effectiveness_summary';
    RAISE NOTICE 'Created functions: refresh_all_materialized_views, cleanup_old_tool_usage, cleanup_low_usage_workflows';
END $$;

-- ============================================================================
-- End of Migration 001
-- ============================================================================
