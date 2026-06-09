"""
ShardRegistry — dataset_id ↔ shard_id 매핑을 관리하는 메타 DB 테이블.

메인 DB(shard_0)에 shard_map 테이블을 두고,
데이터셋 생성 시 일관된 해시(consistent hashing)로 샤드를 배정합니다.
이후 모든 쿼리는 레지스트리를 거쳐 올바른 샤드로 라우팅됩니다.
"""
import hashlib
from functools import lru_cache
from sqlalchemy import Column, Integer, String, DateTime, func, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import select


# ── 메타 DB ORM (shard_map 전용) ──────────────────────────────

class MetaBase(DeclarativeBase):
    pass


class ShardMap(MetaBase):
    """dataset_id → shard_id 매핑 테이블 (메인 DB에 저장)"""
    __tablename__ = "shard_map"

    dataset_id = Column(Integer, primary_key=True)
    shard_id   = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ── 일관된 해시 라우팅 ────────────────────────────────────────

def hash_to_shard(dataset_id: int, shard_count: int) -> int:
    """
    dataset_id를 SHA-256 해시하여 shard_count 내의 샤드 번호 반환.
    동일한 dataset_id는 항상 동일한 샤드로 라우팅됩니다.
    """
    digest = hashlib.sha256(str(dataset_id).encode()).hexdigest()
    return int(digest, 16) % shard_count


# ── ShardRegistry 클래스 ──────────────────────────────────────

class ShardRegistry:
    """
    dataset_id → shard_id 매핑을 메모리 캐시 + DB로 관리.

    - assign(dataset_id): 새 데이터셋에 샤드 배정
    - lookup(dataset_id): 기존 데이터셋의 샤드 반환
    """

    def __init__(self, meta_db_url: str, shard_count: int):
        self.shard_count = shard_count
        self._engine = create_async_engine(meta_db_url, echo=False)
        self._session = async_sessionmaker(
            bind=self._engine, expire_on_commit=False
        )
        self._cache: dict[int, int] = {}  # in-memory 캐시

    async def initialize(self):
        """애플리케이션 시작 시 shard_map 테이블 생성"""
        async with self._engine.begin() as conn:
            await conn.run_sync(MetaBase.metadata.create_all)
        # 기존 매핑 로드
        async with self._session() as s:
            rows = await s.execute(select(ShardMap))
            for row in rows.scalars():
                self._cache[row.dataset_id] = row.shard_id

    async def assign(self, dataset_id: int) -> int:
        """새 데이터셋에 샤드를 배정하고 저장."""
        if dataset_id in self._cache:
            return self._cache[dataset_id]

        shard_id = hash_to_shard(dataset_id, self.shard_count)

        async with self._session() as s:
            # 이미 존재하면 그것을 사용 (race condition 방어)
            existing = await s.get(ShardMap, dataset_id)
            if existing:
                self._cache[dataset_id] = existing.shard_id
                return existing.shard_id

            entry = ShardMap(dataset_id=dataset_id, shard_id=shard_id)
            s.add(entry)
            await s.commit()

        self._cache[dataset_id] = shard_id
        return shard_id

    async def lookup(self, dataset_id: int) -> int | None:
        """dataset_id의 샤드 번호 조회. 없으면 None."""
        if dataset_id in self._cache:
            return self._cache[dataset_id]

        async with self._session() as s:
            row = await s.get(ShardMap, dataset_id)
            if row:
                self._cache[dataset_id] = row.shard_id
                return row.shard_id

        return None

    async def remove(self, dataset_id: int):
        """데이터셋 삭제 시 매핑 제거."""
        self._cache.pop(dataset_id, None)
        async with self._session() as s:
            row = await s.get(ShardMap, dataset_id)
            if row:
                await s.delete(row)
                await s.commit()

    def shard_stats(self) -> dict[int, int]:
        """각 샤드에 배정된 데이터셋 수 반환 (운영 모니터링용)."""
        stats: dict[int, int] = {i: 0 for i in range(self.shard_count)}
        for shard_id in self._cache.values():
            stats[shard_id] = stats.get(shard_id, 0) + 1
        return stats
