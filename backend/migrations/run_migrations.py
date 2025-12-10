"""PostgreSQL migration runner for Level 3c: Procedural & Reflective Memory.

This script runs SQL migrations to initialize PostgreSQL schema for Layers 5 & 7.

Usage:
    # Run all pending migrations
    python -m backend.migrations.run_migrations

    # Run specific migration
    python -m backend.migrations.run_migrations --migration 001

    # Rollback last migration
    python -m backend.migrations.run_migrations --rollback

Environment Variables Required:
    - POSTGRES_URL: PostgreSQL connection URL
      Format: postgresql://user:password@host:port/database
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import asyncpg

from backend.config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Migration directory
MIGRATIONS_DIR = Path(__file__).parent
MIGRATION_TABLE = "schema_migrations"


class MigrationRunner:
    """PostgreSQL migration runner with version tracking."""

    def __init__(self, database_url: str) -> None:
        """Initialize migration runner.

        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
        self.conn: asyncpg.Connection | None = None

    async def __aenter__(self) -> "MigrationRunner":
        """Async context manager entry."""
        self.conn = await asyncpg.connect(self.database_url)
        await self._ensure_migrations_table()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self.conn:
            await self.conn.close()

    async def _ensure_migrations_table(self) -> None:
        """Create schema_migrations table if it doesn't exist."""
        if not self.conn:
            raise RuntimeError("Database connection not established")

        await self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} (
                id SERIAL PRIMARY KEY,
                migration_name VARCHAR(255) NOT NULL UNIQUE,
                applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                checksum VARCHAR(64) NOT NULL
            )
        """
        )
        logger.info(f"Ensured {MIGRATION_TABLE} table exists")

    async def _get_applied_migrations(self) -> set[str]:
        """Get list of already applied migrations.

        Returns:
            Set of migration names that have been applied
        """
        if not self.conn:
            raise RuntimeError("Database connection not established")

        rows = await self.conn.fetch(
            f"SELECT migration_name FROM {MIGRATION_TABLE} ORDER BY id"
        )
        return {row["migration_name"] for row in rows}

    async def _mark_migration_applied(
        self, migration_name: str, checksum: str
    ) -> None:
        """Mark a migration as applied.

        Args:
            migration_name: Name of the migration file
            checksum: SHA256 checksum of migration content
        """
        if not self.conn:
            raise RuntimeError("Database connection not established")

        await self.conn.execute(
            f"""
            INSERT INTO {MIGRATION_TABLE} (migration_name, checksum)
            VALUES ($1, $2)
            ON CONFLICT (migration_name) DO NOTHING
        """,
            migration_name,
            checksum,
        )
        logger.info(f"Marked migration as applied: {migration_name}")

    async def _run_migration(self, migration_file: Path) -> None:
        """Run a single migration file.

        Args:
            migration_file: Path to SQL migration file
        """
        if not self.conn:
            raise RuntimeError("Database connection not established")

        migration_name = migration_file.name
        sql_content = migration_file.read_text()

        # Calculate checksum (simple hash for now)
        import hashlib

        checksum = hashlib.sha256(sql_content.encode()).hexdigest()

        logger.info(f"Running migration: {migration_name}")

        try:
            # Execute migration in a transaction
            async with self.conn.transaction():
                # Execute SQL
                await self.conn.execute(sql_content)

                # Mark as applied
                await self._mark_migration_applied(migration_name, checksum)

            logger.info(f"✅ Successfully applied migration: {migration_name}")

        except Exception as e:
            logger.error(f"❌ Failed to apply migration {migration_name}: {e}")
            raise

    async def run_all_migrations(self) -> None:
        """Run all pending SQL migrations in order."""
        if not self.conn:
            raise RuntimeError("Database connection not established")

        # Get list of applied migrations
        applied = await self._get_applied_migrations()

        # Get list of migration files
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        if not migration_files:
            logger.warning(f"No migration files found in {MIGRATIONS_DIR}")
            return

        logger.info(f"Found {len(migration_files)} migration file(s)")

        # Run pending migrations
        pending_count = 0
        for migration_file in migration_files:
            if migration_file.name not in applied:
                await self._run_migration(migration_file)
                pending_count += 1
            else:
                logger.info(f"⏭️  Skipping already applied: {migration_file.name}")

        if pending_count == 0:
            logger.info("✅ All migrations are up to date!")
        else:
            logger.info(f"✅ Applied {pending_count} migration(s) successfully!")

    async def rollback_last_migration(self) -> None:
        """Rollback the last applied migration.

        Note: This is destructive! Only use in development.
        """
        if not self.conn:
            raise RuntimeError("Database connection not established")

        logger.warning("⚠️  Rollback is not implemented for safety reasons")
        logger.warning(
            "To rollback, manually drop tables and re-run migrations from scratch"
        )
        logger.warning("Development rollback command:")
        logger.warning("  docker exec -it weather-ai-postgres-dev psql -U weather_ai -d weather_ai")
        logger.warning("  DROP TABLE IF EXISTS workflows, tool_usage, reflections CASCADE;")
        logger.warning(f"  DELETE FROM {MIGRATION_TABLE};")


async def main() -> None:
    """Main entry point for migration runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run PostgreSQL migrations")
    parser.add_argument(
        "--migration",
        type=str,
        help="Run specific migration by name (e.g., 001)",
        default=None,
    )
    parser.add_argument(
        "--rollback", action="store_true", help="Rollback last migration (DESTRUCTIVE)"
    )
    args = parser.parse_args()

    # Load settings
    settings = Settings()

    if not settings.POSTGRES_URL:
        logger.error("❌ POSTGRES_URL environment variable not set!")
        logger.error("Set it in .env file or export manually:")
        logger.error(
            "  export POSTGRES_URL=postgresql://weather_ai:weatherai2025@localhost:5432/weather_ai"
        )
        return

    logger.info(f"Connecting to PostgreSQL: {settings.POSTGRES_URL.split('@')[1]}")

    try:
        async with MigrationRunner(str(settings.POSTGRES_URL)) as runner:
            if args.rollback:
                await runner.rollback_last_migration()
            elif args.migration:
                migration_file = MIGRATIONS_DIR / f"{args.migration}_*.sql"
                matching = list(MIGRATIONS_DIR.glob(f"{args.migration}_*.sql"))
                if not matching:
                    logger.error(f"❌ Migration not found: {args.migration}")
                    return
                await runner._run_migration(matching[0])
            else:
                await runner.run_all_migrations()

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
