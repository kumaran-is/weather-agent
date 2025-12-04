# ============================================================================
# Weather AI Agent Service - Dockerfile
# ============================================================================
# Multi-stage build for production deployment
#
# Build stages:
#   1. base: Python 3.13 with uv for dependency management
#   2. dependencies: Install production dependencies
#   3. development: Add dev dependencies (testing, linting)
#   4. production: Minimal runtime image with only production deps
#
# Build commands:
#   docker build -t weather-ai-agent:latest .
#   docker build --target development -t weather-ai-agent:dev .
# ============================================================================

# ============================================================================
# Stage 1: Base Image with Python 3.13 + uv
# ============================================================================
FROM python:3.13.5-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (ultra-fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files and README.md (required by pyproject.toml)
COPY pyproject.toml uv.lock README.md ./

# ============================================================================
# Stage 2: Install Production Dependencies
# ============================================================================
FROM base AS dependencies

# Create virtual environment and install production dependencies
RUN uv sync --frozen --no-dev

# ============================================================================
# Stage 3: Development Image (with dev dependencies)
# ============================================================================
FROM dependencies AS development

# Install dev dependencies (testing, linting, type checking)
RUN uv sync --frozen

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Run with uvicorn (auto-reload enabled for development)
CMD ["uv", "run", "uvicorn", "backend.src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================================================
# Stage 4: Production Image (minimal, production-only deps)
# ============================================================================
FROM base AS production

# Copy only production virtual environment from dependencies stage
COPY --from=dependencies /app/.venv /app/.venv

# Copy application code
COPY backend /app/backend
COPY pyproject.toml uv.lock /app/

# Create non-root user for security
RUN useradd -m -u 1000 weather && \
    chown -R weather:weather /app

USER weather

# Expose FastAPI port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Health check (FastAPI /health endpoint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn (production settings)
CMD ["uvicorn", "backend.src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
