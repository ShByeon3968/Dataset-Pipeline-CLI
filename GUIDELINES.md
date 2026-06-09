# Dataset Pipeline CLI — 개발자 가이드

## 개요

React 프론트엔드를 대신하는 Python CLI 도구입니다.  
백엔드(FastAPI)는 **그대로 유지**하고, 모든 작업을 터미널 스크립트로 수행합니다.  
수동 레이블링 기능은 제거되었으며 **모델 기반 Auto Labeling만** 지원합니다.

---

## 프로젝트 구조

```
Dataset Pipeline CLI/
├── backend/                  # FastAPI 백엔드 (원본과 동일)
│   ├── app/
│   │   ├── main.py
│   │   ├── models/
│   │   ├── routers/
│   │   ├── services/
│   │   │   ├── auto_label_service.py   # YOLO-World
│   │   │   ├── onnx_inference.py       # ONNX 커스텀 모델
│   │   │   ├── detectors/              # RF-DETR, DEIM-v2 어댑터
│   │   │   └── ...
│   │   └── sharding/
│   └── requirements.txt
│
├── cli/                      # CLI 클라이언트 레이어
│   ├── config.py             # 설정 (.pipeline.toml / 환경변수)
│   ├── client.py             # httpx API 클라이언트
│   ├── display.py            # rich 터미널 출력
│   └── commands/
│       ├── dataset.py        # pipeline dataset *
│       ├── upload.py         # pipeline upload *
│       ├── autolabel.py      # pipeline autolabel *
│       ├── analysis.py       # pipeline analysis *
│       ├── refinement.py     # pipeline refinement *
│       ├── export.py         # pipeline export *
│       └── version.py        # pipeline version *
│
├── pipeline.py               # CLI 진입점
├── requirements.txt          # CLI 의존성 (typer, rich, httpx)
├── .pipeline.toml.example    # 설정 파일 예시
├── docker-compose.yml        # db + backend (frontend 제거됨)
└── docker-compose.sharded.yml
```

---

## 빠른 시작

### 1. 서버 실행

```bash
docker compose up -d
```

### 2. CLI 환경 설정

```bash
# Python 3.11+ 권장
pip install -r requirements.txt

# 설정 파일 복사 (선택)
cp .pipeline.toml.example .pipeline.toml
# base_url 등 필요 시 수정
```

### 3. 연결 확인

```bash
python pipeline.py health
```

---

## 주요 워크플로우

### 데이터셋 생성 → 업로드 → 자동 라벨링 → 내보내기

```bash
# 1. 데이터셋 생성
python pipeline.py dataset create --name "MyDataset" --desc "자율주행 데이터"

# 2. 이미지 업로드 (디렉토리)
python pipeline.py upload images --dataset-id 1 --path ./images/

# 또는 ZIP 업로드 (train/val/test split 포함 가능)
python pipeline.py upload zip --dataset-id 1 --path ./dataset.zip

# 3. 자동 라벨링 (YOLO-World)
python pipeline.py autolabel run \
  --dataset-id 1 \
  --model yolo-world \
  --prompts "person,car,truck,bicycle,traffic light" \
  --conf 0.3

# 3-b. ONNX 커스텀 모델 사용
python pipeline.py autolabel run \
  --dataset-id 1 \
  --model onnx \
  --onnx-model-id 2 \
  --conf 0.25 --iou 0.45

# 4. 분석
python pipeline.py analysis stats --dataset-id 1
python pipeline.py analysis duplicates --dataset-id 1

# 5. 정제
python pipeline.py refinement errors --dataset-id 1
python pipeline.py refinement filter-bbox --dataset-id 1 --min-area 0.001 --execute

# 6. 버전 스냅샷
python pipeline.py version create --dataset-id 1 --name "v1.0" --branch main

# 7. 내보내기
python pipeline.py export yolo --dataset-id 1 --out ./output/
python pipeline.py export coco --dataset-id 1 --out ./output/
```

---

## 설정 우선순위

```
환경변수  >  .pipeline.toml (현재 디렉토리)  >  ~/.pipeline.toml  >  내장 기본값
```

| 환경변수 | 설명 | 기본값 |
|----------|------|--------|
| `PIPELINE_BASE_URL` | 백엔드 주소 | `http://localhost:8000` |
| `PIPELINE_TIMEOUT`  | HTTP 타임아웃(초) | `120` |

---

## 모든 명령어 도움말

```bash
python pipeline.py --help
python pipeline.py dataset --help
python pipeline.py upload --help
python pipeline.py autolabel --help
python pipeline.py analysis --help
python pipeline.py refinement --help
python pipeline.py export --help
python pipeline.py version --help
```

---

## 제거된 기능

| 기능 | 이유 |
|------|------|
| React 프론트엔드 (`frontend/`) | CLI로 대체 |
| 수동 레이블링 (`POST /annotations`) | 모델 기반 자동 라벨링만 사용 |
| 브라우저 캔버스 편집기 | 불필요 |
