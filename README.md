# YOLO26 건축 도면 객체 탐지 시스템

이 프로젝트는 YOLO26 모델을 사용하여 건축 도면에서 화장실 관련 객체(변기, 세면대, 싱크대, 욕조, 가스레인지)를 탐지하는 시스템입니다.

## 🚀 주요 기능

- **YOLO26 모델**: 최신 YOLO26 아키텍처 사용 (NMS-free, End-to-End)
- **분산 학습**: RTX 4090 4장을 활용한 멀티 GPU 학습
- **MuSGD Optimizer**: YOLO26의 새로운 하이브리드 옵티마이저
- **W&B 통합**: 실시간 학습 모니터링 및 Fitness 스코어 추적
- **단계별 학습**: 효율적인 GPU 리소스 활용
- **고성능 추론**: 학습된 모델로 커스텀 이미지 추론

## 📦 빠른 시작

### 로컬 환경

```bash
# 1. 클론 및 환경 설정
git clone https://github.com/yourname/wishket.git
cd wishket
cp .env.example .env
# .env 파일에 WANDB_API_KEY 설정

# 2. 의존성 설치
uv sync  # 또는 pip install -r requirements.txt

# 3. 학습
./run_all_training_cli.sh

# 4. 추론
uv run code/infer_yolo26.py --model_size s --images_dir /path/to/images
```

### Docker 환경

```bash
# 1. 클론 및 환경 설정
git clone https://github.com/yourname/wishket.git
cd wishket
cp .env.example .env
# .env 파일에 WANDB_API_KEY 설정

# 2. Docker 실행
docker-compose up -d
docker-compose exec yolo26-train bash

# 3. 컨테이너 내에서
./run_all_training_cli.sh  # 학습
uv run code/infer_yolo26.py --model_size s --images_dir /path/to/images  # 추론
```

자세한 Docker 사용법은 [DOCKER_GUIDE.md](DOCKER_GUIDE.md)를 참조하세요.

### 필요 조건

**로컬:** Python 3.11+, CUDA 12.4+, GPU  
**Docker:** Docker 20.10+, Docker Compose 2.0+, NVIDIA Docker Runtime

## 🎯 타겟 클래스 (JSON → YOLO 변환)

| JSON 클래스 ID | YOLO 클래스 ID | 클래스 이름 | 설명 |
|---------------|----------------|------------|------|
| 4 | 0 | toilet | 객체_변기 |
| 5 | 1 | washstand | 객체_세면대 |
| 6 | 2 | sink | 객체_싱크대 |
| 7 | 3 | bathtub | 객체_욕조 |
| 8 | 4 | gas_stove | 객체_가스레인지 |

## 🔧 사용법

### 학습

```bash
# 로컬
./run_all_training_cli.sh

# Docker
docker-compose exec yolo26-train ./run_all_training_cli.sh
```

### 추론

```bash
# 로컬
uv run code/infer_yolo26.py --model_size s --images_dir /path/to/images

# Docker (컨테이너 내)
uv run code/infer_yolo26.py --model_size s --images_dir /path/to/images

# Docker (외부에서 실행)
docker run --rm --gpus all \
  -v /path/to/images:/workspace/input:ro \
  -v $(pwd)/inference_results:/workspace/inference_results \
  yolo26-construction:latest \
  uv run code/infer_yolo26.py --model_size s --images_dir /workspace/input
```

### 추론 옵션

```bash
--model_size s|m|l|x    # 모델 크기
--images_dir <path>     # 이미지 디렉토리
--gpu <number>          # GPU 번호 (기본값: 0)
--conf <float>          # 신뢰도 임계값 (기본값: 0.25)
--iou <float>           # IoU 임계값 (기본값: 0.45)
--output_dir <path>     # 출력 디렉토리 (생략시 자동 생성)
```

## 📊 데이터셋 구조

```
dataset/
├── data.yml            # YOLO 설정 파일
├── train/
│   ├── images/         # 학습 이미지 (.png)
│   └── labels/         # YOLO 형식 라벨 (.txt)
└── valid/
    ├── images/         # 검증 이미지 (.png)
    └── labels/         # YOLO 형식 라벨 (.txt)
```

## 🏗️ 모델 아키텍처

### YOLO26의 주요 개선사항
- **DFL 제거**: 내보내기 간소화 및 엣지 호환성 향상
- **End-to-End NMS-Free**: 후처리 없는 직접 예측
- **ProgLoss + STAL**: 소형 객체 탐지 정확도 향상
- **MuSGD Optimizer**: SGD + Muon 하이브리드 옵티마이저
- **43% 빠른 CPU 추론**: 엣지 디바이스 최적화

## 📈 학습 파이프라인

1. **데이터 변환**: JSON → YOLO 형식 변환 (클래스 매핑 4-8 → 0-4)
2. **단계별 학습**: 
   - 1단계: s(GPU 0,1) + m(GPU 2,3) 동시 학습
   - 2단계: l(GPU 0,1,2,3) + x(GPU 0,1,2,3) 순차 학습
3. **W&B 모니터링**: 실시간 fitness 스코어 및 메트릭 추적
4. **자동 조기 종료**: patience 기반 최적화

## 🎛️ 하이퍼파라미터

### 기본 설정
- **Learning Rate**: 0.01 (초기) → 0.001 (최종)
- **Momentum**: 0.937
- **Weight Decay**: 0.0005
- **Warmup Epochs**: 3
- **Image Size**: 640×640
- **Augmentation**: HSV, Translate, Scale, Flip

### 모델 크기별 권장 설정
| 모델 | 배치 크기 | 메모리 사용량 | 학습 시간 |
|------|----------|-------------|----------|
| yolo26n | 64-128 | ~6GB | 빠름 |
| yolo26s | 32-64 | ~8GB | 보통 |
| yolo26m | 16-32 | ~12GB | 느림 |
| yolo26l | 8-16 | ~16GB | 매우 느림 |
| yolo26x | 4-8 | ~20GB+ | 극도로 느림 |

## 📋 결과 분석

### 출력 메트릭
- **mAP@0.5**: IoU 0.5에서의 평균 정밀도
- **mAP@0.5:0.95**: IoU 0.5~0.95 범위 평균 정밀도
- **Fitness Score**: 0.1*mAP50 + 0.9*mAP50-95 (W&B 추적)
- **Precision**: 정밀도
- **Recall**: 재현율

### 추론 결과
- **객체 탐지 이미지**: 바운딩 박스 + 라벨 + 신뢰도
- **저장 형식**: JPG/PNG 이미지
- **커스텀 설정**: 폰트 크기, 선 두께, 신뢰도/IoU 임계값

## 🔧 핵심 파일 및 스크립트

```
프로젝트 디렉토리
├── setup.sh                      # 🚀 초기 설정 (로컬/Docker 공통)
├── docker-train.sh               # 🐳 Docker 학습 시작
├── docker-infer.sh               # 🐳 Docker 추론 실행
├── run_all_training_cli.sh       # 단계별 학습 (로컬)
├── Dockerfile                    # Docker 이미지 정의
├── docker-compose.yml            # Docker Compose 설정
├── .dockerignore                 # Docker 제외 파일
├── .env.example                  # 환경 변수 템플릿
├── code/
│   ├── yolo26_train.py           # 간단한 학습
│   ├── yolo26_train_cli.py       # 고급 학습 (W&B)
│   ├── infer_yolo26.py           # 추론
│   ├── convert_json_to_yolo_ultralytics.py  # JSON→YOLO 변환
│   └── README.md                 # 상세 설명
├── dataset/
│   ├── data.yml                  # 데이터셋 설정
│   ├── train/
│   │   ├── images/               # 학습 이미지 (볼륨 마운트)
│   │   └── labels/               # 학습 라벨
│   └── valid/
│       ├── images/               # 검증 이미지 (볼륨 마운트)
│       └── labels/               # 검증 라벨
├── fine_tuning_weights/          # 학습된 모델
├── results/                      # 학습 결과 (로컬/Docker 공유)
├── training_logs/                # 학습 로그 (로컬/Docker 공유)
└── inference_results/            # 추론 결과 (로컬/Docker 공유)
```

### 스크립트 설명

| 스크립트 | 환경 | 설명 |
|---------|------|------|
| `setup.sh` | 로컬/Docker | 초기 설정 (.env, 디렉토리, 권한) |
| `run_all_training_cli.sh` | 로컬 | 단계별 학습 실행 |
| `docker-train.sh` | Docker | Docker Compose로 학습 시작 |
| `docker-infer.sh` | Docker | Docker로 추론 실행 |

## 🚨 문제 해결

### GPU 메모리 부족
```bash
# 배치 크기 줄이기
python train_yolo.py --batch_size 8

# 더 작은 모델 사용
python train_yolo.py --model_size n
```

### CUDA 오류
```bash
# GPU 상태 확인
nvidia-smi

# CUDA 캐시 정리
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### W&B 연결 문제
```bash
# W&B 비활성화
python train_yolo.py --disable_wandb

# W&B 재로그인
wandb login
```

### 추론 문제 해결
```bash
# 모델 상태 확인
ls -la results/*/weights/best.pt

# 이미지 디렉토리 확인
ls -la /path/to/images/*.{jpg,png}

# 폰트 크기 조절
uv run code/infer_yolo26.py --font_size 20
```

## 📞 지원 및 문의

- 로그 모니터링: `tail -f training_logs/*.log`
- GPU 모니터링: `nvidia-smi -l 1`
- W&B 모니터링: 프로젝트 `yolo26-construction`
- Code 상세 설명: [code/README.md](code/README.md)

## 🐳 Docker 배포

Docker를 사용하면 환경 설정 없이 바로 실행할 수 있습니다. 자세한 내용은 [DOCKER_GUIDE.md](DOCKER_GUIDE.md)를 참조하세요.

---

**주의사항**: 
- 모든 경로는 상대 경로로 설정되어 있어 프로젝트 루트 내 어디서든 실행 가능
- 학습 전 GPU 메모리 상태 확인 (`nvidia-smi`)
- W&B는 `.env` 파일에 API 키 설정 시 자동 연결

