# 🐳 Docker 배포 가이드

## 📋 배포 전 체크리스트

### 1. ✅ 완료된 수정 사항
- [x] 절대 경로 → 상대 경로 변경
- [x] 사용자명 노출 제거
- [x] GPU 자동 감지 기능 추가
- [x] 파일명 통일 (`yolo26_` 접두사)
- [x] `.env.example` 생성
- [x] `requirements.txt` 백업 생성
- [x] `original_code` 폴더 제거

### 2. 📦 프로젝트 구조
```
wishket/
├── Dockerfile              # Docker 이미지 정의
├── docker-compose.yml      # Docker Compose 설정
├── .dockerignore          # 이미지에서 제외할 파일
├── .env.example           # 환경 변수 템플릿
├── pyproject.toml         # uv 프로젝트 설정
├── requirements.txt       # pip 호환 의존성 (백업)
├── code/
│   ├── yolo26_train.py          # 간단한 학습
│   ├── yolo26_train_cli.py      # 고급 학습 (W&B)
│   ├── infer_yolo26.py          # 추론
│   └── convert_json_to_yolo_ultralytics.py
├── dataset/
│   ├── data.yml           # 데이터셋 설정 (상대 경로)
│   ├── train/
│   │   ├── labels/*.txt   # 라벨 (포함)
│   │   ├── labels.cache   # 캐시 (포함)
│   │   └── images/        # 이미지 (볼륨 마운트)
│   └── valid/
│       ├── labels/*.txt
│       ├── labels.cache
│       └── images/
└── fine_tuning_weights/   # 학습된 모델 가중치
    ├── yolo26s_best.pt
    ├── yolo26m_best.pt
    ├── yolo26l_best.pt
    └── yolo26x_best.pt
```

---

## 🚀 Docker 이미지 빌드 및 실행

### 방법 1: Docker CLI 사용

#### 1️⃣ 이미지 빌드
```bash
cd /path/to/wishket  # 프로젝트를 다운로드한 디렉토리로 이동

# 이미지 빌드 (5-10분 소요)
docker build -t yolo26-construction:latest .

# 빌드 확인
docker images | grep yolo26
```

#### 2️⃣ 컨테이너 실행
```bash
# 모든 GPU 사용, 볼륨 마운트
docker run --gpus all -it \
  -v $(pwd)/dataset/train/images:/workspace/dataset/train/images:ro \
  -v $(pwd)/dataset/valid/images:/workspace/dataset/valid/images:ro \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/training_logs:/workspace/training_logs \
  -v $(pwd)/inference_results:/workspace/inference_results \
  --env-file .env \
  --name yolo26-train \
  yolo26-construction:latest
```

#### 3️⃣ 학습 시작
```bash
# 컨테이너 내부에서
./run_all_training_cli.sh
```

---

### 방법 2: Docker Compose 사용 (권장)

#### 1️⃣ 서비스 시작
```bash
# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

#### 2️⃣ 컨테이너 접속
```bash
docker-compose exec yolo26-train bash
```

#### 3️⃣ 학습/추론 실행
```bash
# 전체 학습
./run_all_training_cli.sh

# 개별 모델 학습
uv run python code/yolo26_train_cli.py --model_size x

# 추론
uv run python code/infer_yolo26.py --model_size s --images_dir dataset/valid/images
```

#### 4️⃣ 서비스 중지
```bash
docker-compose down
```

---

## 📤 Docker 이미지 배포

### 1. Docker Hub에 업로드

#### 로그인
```bash
docker login
```

#### 태그 지정
```bash
docker tag yolo26-construction:latest your-username/yolo26-construction:latest
docker tag yolo26-construction:latest your-username/yolo26-construction:v1.0
```

#### 푸시
```bash
docker push your-username/yolo26-construction:latest
docker push your-username/yolo26-construction:v1.0
```

### 2. 다른 사용자의 사용법

#### 이미지 다운로드
```bash
docker pull your-username/yolo26-construction:latest
```

#### .env 설정
```bash
# .env.example을 .env로 복사
cp .env.example .env

# WANDB_API_KEY 입력
nano .env
```

#### 실행
```bash
# Docker Compose 사용
docker-compose up -d
docker-compose exec yolo26-train bash

# 또는 Docker CLI
docker run --gpus all -it \
  --env-file .env \
  -v $(pwd)/dataset/train/images:/workspace/dataset/train/images:ro \
  -v $(pwd)/dataset/valid/images:/workspace/dataset/valid/images:ro \
  -v $(pwd)/results:/workspace/results \
  your-username/yolo26-construction:latest
```

---

## 🔍 트러블슈팅

### GPU 인식 안 됨
```bash
# NVIDIA Docker 런타임 확인
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# docker-compose에서 GPU 확인
docker-compose exec yolo26-train nvidia-smi
```

### 볼륨 마운트 문제
```bash
# 절대 경로 사용
docker run ... -v /full/path/to/dataset:/workspace/dataset ...

# 권한 확인
ls -la dataset/train/images
```

### uv 명령어 안 됨
```bash
# 컨테이너 내부에서
export PATH="/root/.local/bin:$PATH"
uv --version
```

---

## 💡 최적화 팁

### 이미지 크기 줄이기
- 불필요한 파일 `.dockerignore`에 추가
- Multi-stage build 사용
- 레이어 최적화

### 빌드 속도 향상
- Docker BuildKit 사용: `DOCKER_BUILDKIT=1 docker build ...`
- 캐시 활용: 자주 변경되는 파일은 나중에 COPY

### 보안
- `.env` 파일을 절대 이미지에 포함하지 말 것
- 비밀번호는 Docker secrets 사용
- 이미지 취약점 스캔: `docker scan yolo26-construction:latest`

---

## 📚 추가 자료

- [Docker 공식 문서](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Docker Compose GPU 지원](https://docs.docker.com/compose/gpu-support/)
