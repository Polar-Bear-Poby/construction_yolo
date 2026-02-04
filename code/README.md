# YOLO26 건축 도면 객체 탐지 시스템

이 프로젝트는 YOLO26 모델을 사용하여 건축 도면에서 화장실 관련 객체(변기, 세면대, 싱크대, 욕조, 가스레인지)를 탐지하는 시스템입니다.

## 🚀 주요 기능

- **YOLO26 모델**: 최신 YOLO26 아키텍처 사용 (NMS-free, End-to-End)
- **분산 학습**: RTX 4090 4장을 활용한 멀티 GPU 학습
- **MuSGD Optimizer**: YOLO26의 새로운 하이브리드 옵티마이저
- **W&B 통합**: 실시간 학습 모니터링 및 Fitness 스코어 추적
- **단계별 학습**: 효율적인 GPU 리소스 활용

## 📦 설치 및 환경 설정

### 1. 필요 조건
- Python 3.11+
- CUDA 지원 GPU (RTX 4090 x4 권장)
- uv 패키지 매니저

### 2. 설치
```bash
# 프로젝트 디렉토리로 이동
cd /home/themiraclesoft/wishket

# 의존성 설치 (이미 완료됨)
# uv add ultralytics

# 실행 권한 부여
chmod +x run_all_training_cli.sh
```

## 🎯 타겟 클래스 (JSON → YOLO 변환)

| JSON 클래스 ID | YOLO 클래스 ID | 클래스 이름 | 설명 |
|---------------|----------------|------------|------|
| 4 | 0 | toilet | 객체_변기 |
| 5 | 1 | washstand | 객체_세면대 |
| 6 | 2 | sink | 객체_싱크대 |
| 7 | 3 | bathtub | 객체_욕조 |
| 8 | 4 | gas_stove | 객체_가스레인지 |

## 🔧 사용법

### 1. 전체 모델 단계별 학습
```bash
# s+m 동시 → l+x 순차 학습
./run_all_training_cli.sh
```

### 2. 개별 모델 학습
```bash
# YOLO26s 학습
python train_yolo26_cli.py --model_size s --epochs 200 --batch_size 64

# YOLO26m 학습 (GPU 0,1 사용)
python train_yolo26_cli.py --model_size m --epochs 300 --device "0,1"
```

### 3. JSON 데이터 변환
```bash
# COCO JSON을 YOLO 형식으로 변환
python code/convert_json_to_yolo_ultralytics.py
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

## 🔧 핵심 파일 구조

```
/home/themiraclesoft/wishket/
├── run_all_training_cli.sh        # 메인 단계별 학습 스크립트
├── train_yolo26_cli.py            # 개별 모델 학습 스크립트
├── code/
│   ├── convert_json_to_yolo_ultralytics.py  # JSON→YOLO 변환
│   ├── yolo_train.py              # 학습 유틸리티
│   └── README.md                  # 이 파일
├── dataset/                       # YOLO 데이터셋
└── yolo26*.pt                     # 사전 훈련된 모델들
```

## 🚨 문제 해결

### GPU 메모리 부족
```bash
# 배치 크기 줄이기
python train_yolo26_cli.py --batch_size 8

# 더 작은 모델 사용
python train_yolo26_cli.py --model_size n
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
python train_yolo26_cli.py --disable_wandb

# W&B 재로그인
wandb login
```

## 📞 지원 및 문의

- 프로젝트 위치: `/home/themiraclesoft/wishket/`
- 로그 모니터링: `tail -f training_logs/*.log`
- GPU 모니터링: `nvidia-smi -l 1`
- W&B 모니터링: 프로젝트 `yolo26-construction`

---

**주의사항**: 
- 학습 전 GPU 메모리 상태를 확인하세요
- nohup 실행 시 로그 파일 위치를 확인하세요
- W&B는 기본적으로 자동 연결됩니다
