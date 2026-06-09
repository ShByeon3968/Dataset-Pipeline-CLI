from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = "Dataset Management Solution"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "postgres"
    db_password: str = "postgres"
    db_name: str = "dataset_pipeline"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # File storage
    # Docker 볼륨 마운트로 NAS 경로를 /app/data/uploads 에 연결하면
    # 아래 기본값 그대로 사용해도 NAS에 저장됩니다.
    uploads_dir: str = "./data/uploads"
    exports_dir: str = "./data/exports"
    embeddings_dir: str = "./data/embeddings"
    models_dir: str = "./data/models"

    # CORS - 쉼표 구분 문자열로 설정
    # 예) CORS_ORIGINS=http://192.168.1.10:8080,http://192.168.1.20:8080
    cors_origins: str = (
        "http://localhost:5173,"
        "http://localhost:3000,"
        "http://localhost:8080,"
        "http://localhost"
    )
    cors_origins_raw: str | None = None

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS_ORIGINS 또는 CORS_ORIGINS_RAW 환경변수를 list[str] 로 반환."""
        v = self.cors_origins_raw or self.cors_origins
        v = v.strip()
        if not v:
            return ["*"]
        if v.startswith("["):
            import json
            return json.loads(v)
        return [o.strip() for o in v.split(",") if o.strip()]

    # Roboflow
    roboflow_api_key: str = ""

    # Embedding model: "resnet50" | "clip"
    embedding_model: str = "resnet50"


@lru_cache
def get_settings() -> Settings:
    return Settings()
