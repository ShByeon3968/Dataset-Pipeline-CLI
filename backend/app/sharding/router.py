from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession, AsyncEngine,
    create_async_engine, async_sessionmaker,
)
from app.sharding.config import ShardConfig, build_shard_configs, PARTITION_COUNT
from app.sharding.registry import ShardRegistry
from app.database import Base

# Idempotent migrations for images table
_IMAGE_MIGRATIONS = [
    "ALTER TABLE images ADD COLUMN IF NOT EXISTS split VARCHAR(10)",
]

# Idempotent migrations for annotations table (auto-label fields)
_ANNOTATION_MIGRATIONS = [
    "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS is_auto_generated BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS confidence FLOAT",
    "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS source_prompt TEXT",
    "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS auto_label_run_id INTEGER",
    "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS quality_flag VARCHAR(20)",
]

# Idempotent migrations for auto_label_runs table
_RUN_MIGRATIONS = [
    "ALTER TABLE auto_label_runs ADD COLUMN IF NOT EXISTS text_prompts TEXT",
    "ALTER TABLE auto_label_runs ADD COLUMN IF NOT EXISTS onnx_model_id INTEGER",
]


class ShardRouter:

    def __init__(self):
        self._engines: dict[int, AsyncEngine] = {}
        self._sessions: dict[int, async_sessionmaker] = {}
        self._configs: list[ShardConfig] = []
        self.registry: ShardRegistry | None = None

    async def initialize(self):
        self._configs = build_shard_configs()
        shard_count = len(self._configs)

        for cfg in self._configs:
            engine = create_async_engine(
                cfg.db_url,
                echo=False,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
            )
            self._engines[cfg.shard_id] = engine
            self._sessions[cfg.shard_id] = async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                for stmt in _IMAGE_MIGRATIONS + _ANNOTATION_MIGRATIONS + _RUN_MIGRATIONS:
                    try:
                        await conn.execute(text(stmt))
                    except Exception:
                        pass

        self.registry = ShardRegistry(
            meta_db_url=self._configs[0].db_url,
            shard_count=shard_count,
        )
        await self.registry.initialize()

    async def get_session_for_dataset(self, dataset_id: int) -> AsyncSession:
        if self.registry is None:
            raise RuntimeError("ShardRouter not initialized")
        shard_id = await self.registry.lookup(dataset_id)
        if shard_id is None:
            shard_id = await self.registry.assign(dataset_id)
        session_factory = self._sessions.get(shard_id)
        if session_factory is None:
            raise ValueError(f"Shard {shard_id} not found")
        return session_factory()

    async def assign_dataset(self, dataset_id: int) -> int:
        if self.registry is None:
            raise RuntimeError("ShardRouter not initialized")
        return await self.registry.assign(dataset_id)

    async def remove_dataset(self, dataset_id: int):
        if self.registry:
            await self.registry.remove(dataset_id)

    def get_meta_session(self) -> AsyncSession:
        return self._sessions[0]()

    @property
    def shard_count(self) -> int:
        return len(self._configs)

    def stats(self) -> dict:
        return {
            "shard_count": self.shard_count,
            "partition_count": PARTITION_COUNT,
            "distribution": self.registry.shard_stats() if self.registry else {},
        }

    async def close(self):
        for engine in self._engines.values():
            await engine.dispose()


shard_router = ShardRouter()


async def get_sharded_db(dataset_id: int):
    session = await shard_router.get_session_for_dataset(dataset_id)
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_meta_db():
    session = shard_router.get_meta_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
