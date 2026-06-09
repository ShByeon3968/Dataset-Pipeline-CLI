-- ============================================================
-- Phase 1: images / annotations 테이블 HASH 파티셔닝
-- dataset_id 기준으로 8개 파티션으로 분할
-- ============================================================
-- 실행 전: 기존 테이블 백업 필요
-- psql -U postgres -d dataset_pipeline -f partitions.sql
-- ============================================================

BEGIN;

-- ── images 파티셔닝 ──────────────────────────────────────────

-- 1. 기존 데이터 임시 저장
CREATE TABLE images_backup AS SELECT * FROM images;

-- 2. 기존 테이블 삭제 (FK cascade)
DROP TABLE images CASCADE;

-- 3. 파티션 부모 테이블 생성
CREATE TABLE images (
    id          SERIAL,
    dataset_id  INTEGER NOT NULL,
    filename    VARCHAR(500) NOT NULL,
    filepath    VARCHAR(1000) NOT NULL,
    width       INTEGER,
    height      INTEGER,
    format      VARCHAR(20),
    file_hash   VARCHAR(64),
    phash       VARCHAR(64),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, dataset_id),            -- 파티션 키 포함 필수
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
) PARTITION BY HASH (dataset_id);

-- 4. 파티션 8개 생성
CREATE TABLE images_p0 PARTITION OF images FOR VALUES WITH (MODULUS 8, REMAINDER 0);
CREATE TABLE images_p1 PARTITION OF images FOR VALUES WITH (MODULUS 8, REMAINDER 1);
CREATE TABLE images_p2 PARTITION OF images FOR VALUES WITH (MODULUS 8, REMAINDER 2);
CREATE TABLE images_p3 PARTITION OF images FOR VALUES WITH (MODULUS 8, REMAINDER 3);
CREATE TABLE images_p4 PARTITION OF images FOR VALUES WITH (MODULUS 8, REMAINDER 4);
CREATE TABLE images_p5 PARTITION OF images FOR VALUES WITH (MODULUS 8, REMAINDER 5);
CREATE TABLE images_p6 PARTITION OF images FOR VALUES WITH (MODULUS 8, REMAINDER 6);
CREATE TABLE images_p7 PARTITION OF images FOR VALUES WITH (MODULUS 8, REMAINDER 7);

-- 5. 인덱스 (각 파티션에 자동 적용됨)
CREATE INDEX idx_images_dataset_id ON images (dataset_id);
CREATE INDEX idx_images_file_hash  ON images (file_hash);

-- 6. 데이터 복원
INSERT INTO images SELECT * FROM images_backup;
DROP TABLE images_backup;


-- ── annotations 파티셔닝 ─────────────────────────────────────

CREATE TABLE annotations_backup AS
    SELECT a.* FROM annotations a
    JOIN images i ON a.image_id = i.id;

DROP TABLE annotations CASCADE;

CREATE TABLE annotations (
    id               SERIAL,
    image_id         INTEGER NOT NULL,
    dataset_id       INTEGER NOT NULL,       -- 파티션 키 (역정규화)
    class_id         INTEGER,
    bbox_x           FLOAT,
    bbox_y           FLOAT,
    bbox_w           FLOAT,
    bbox_h           FLOAT,
    segmentation     TEXT,
    annotation_type  VARCHAR(20) DEFAULT 'bbox',
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, dataset_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
) PARTITION BY HASH (dataset_id);

CREATE TABLE annotations_p0 PARTITION OF annotations FOR VALUES WITH (MODULUS 8, REMAINDER 0);
CREATE TABLE annotations_p1 PARTITION OF annotations FOR VALUES WITH (MODULUS 8, REMAINDER 1);
CREATE TABLE annotations_p2 PARTITION OF annotations FOR VALUES WITH (MODULUS 8, REMAINDER 2);
CREATE TABLE annotations_p3 PARTITION OF annotations FOR VALUES WITH (MODULUS 8, REMAINDER 3);
CREATE TABLE annotations_p4 PARTITION OF annotations FOR VALUES WITH (MODULUS 8, REMAINDER 4);
CREATE TABLE annotations_p5 PARTITION OF annotations FOR VALUES WITH (MODULUS 8, REMAINDER 5);
CREATE TABLE annotations_p6 PARTITION OF annotations FOR VALUES WITH (MODULUS 8, REMAINDER 6);
CREATE TABLE annotations_p7 PARTITION OF annotations FOR VALUES WITH (MODULUS 8, REMAINDER 7);

CREATE INDEX idx_annotations_dataset_id ON annotations (dataset_id);
CREATE INDEX idx_annotations_image_id   ON annotations (image_id, dataset_id);

-- 데이터 복원 (dataset_id 채우기)
INSERT INTO annotations (id, image_id, dataset_id, class_id,
    bbox_x, bbox_y, bbox_w, bbox_h, segmentation, annotation_type,
    created_at, updated_at)
SELECT
    a.id, a.image_id, i.dataset_id, a.class_id,
    a.bbox_x, a.bbox_y, a.bbox_w, a.bbox_h,
    a.segmentation, a.annotation_type,
    a.created_at, a.updated_at
FROM annotations_backup a
JOIN images i ON a.image_id = i.id;

DROP TABLE annotations_backup;

COMMIT;

-- ── 확인 쿼리 ─────────────────────────────────────────────────
-- SELECT tableoid::regclass, count(*) FROM images GROUP BY 1;
-- SELECT tableoid::regclass, count(*) FROM annotations GROUP BY 1;
