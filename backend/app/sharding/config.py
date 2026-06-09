"""
샤딩 설정
- Phase 1 (단일 DB): SHARD_COUNT > 0, 모든 샤드 URL이 동일한 DB
- Phase 2 (멀티 DB): SHARD_URLS에 각 샤드별 DB URL 지정
"""
from dataclasses import dataclass, field
from app.core.config import get_settings

# 파티션 수 (PostgreSQL PARTITION BY HASH 개수)
# 2의 거듭제곱 권장: 4, 8, 16, 32
PARTITION_COUNT = 8


@dataclass
class ShardConfig:
    """샤드 1개의 설정"""
    shard_id: int
    db_url: str          # asyncpg URL
    sync_db_url: str     # psycopg2 URL (Alembic용)
    weight: int = 1      # 일관된 해시 가중치


def build_shard_configs() -> list[ShardConfig]:
    """
    환경변수에서 샤드 설정을 읽어 반환.

    단일 DB (Phase 1):
      SHARD_COUNT=1  → 기존 DB URL 하나만 사용

    멀티 DB (Phase 2):
      SHARD_0_DB_HOST, SHARD_1_DB_HOST, ... 로 각 샤드 설정
    """
    settings = get_settings()
    import os

    shard_count = int(os.getenv("SHARD_COUNT", "1"))

    configs = []
    for i in range(shard_count):
        prefix = f"SHARD_{i}_"
        host     = os.getenv(f"{prefix}DB_HOST",     settings.db_host)
        port     = os.getenv(f"{prefix}DB_PORT",     str(settings.db_port))
        user     = os.getenv(f"{prefix}DB_USER",     settings.db_user)
        password = os.getenv(f"{prefix}DB_PASSWORD", settings.db_password)
        dbname   = os.getenv(f"{prefix}DB_NAME",     settings.db_name)

        async_url = (
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{dbname}"
        )
        sync_url = (
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        )
        configs.append(ShardConfig(
            shard_id=i,
            db_url=async_url,
            sync_db_url=sync_url,
        ))

    return configs
