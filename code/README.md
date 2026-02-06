# Code 폴더 상세 가이드

이 폴더에는 YOLO26 프로젝트의 핵심 코드들이 포함되어 있습니다.

## 📁 파일 구조

```
code/
├── infer_yolo26.py                     # 🎯 학습된 모델 추론 스크립트
├── convert_json_to_yolo_ultralytics.py # 🔄 JSON → YOLO 데이터 변환
├── yolo_train.py                       # 🏋️ 학습 유틸리티 함수들
├── yolo26*.pt                          # 📦 사전훈련된 YOLO26 모델들
└── README.md                           # 📖 이 문서
```

## 🎯 infer_yolo26.py - 추론 스크립트

학습된 YOLO26 모델을 사용하여 커스텀 이미지들을 추론하는 메인 스크립트입니다.

### 주요 기능
- **단일 모델 추론**: s, m, l, x 중 하나 선택하여 추론
- **자동 출력 폴더 생성**: 타임스탬프 기반 폴더 자동 생성
- **커스터마이징 가능**: 폰트 크기, confidence, IoU threshold 조절
- **배치 처리**: 폴더 내 모든 이미지 자동 처리

### 사용자 설정 변수 (코드 상단)
```python
# ===== 사용자 설정 변수들 =====
DEFAULT_WEIGHTS_DIR = 'fine_tuning_weights'  # 기본 가중치 폴더
DEFAULT_CONF = 0.25          # Confidence threshold
DEFAULT_IOU = 0.45           # IoU threshold  
DEFAULT_FONT_SIZE = 16       # 라벨 폰트 크기
DEFAULT_LINE_WIDTH = 2       # 바운딩 박스 선 두께
DEFAULT_GPU_DEVICE = '0'     # 사용할 GPU 디바이스
```

### 사용법
```bash
# 기본 사용 (필수: 모델 크기, 이미지 디렉토리)
uv run code/infer_yolo26.py --model_size s --images_dir /path/to/images

# 가중치 폴더 지정하여 추론 (멀티 GPU 병렬 처리)
uv run code/infer_yolo26.py \
    --model_size l \
    --images_dir /path/to/images \
    --weights_dir /path/to/weights \
    --gpu 2

# 모든 옵션 사용
uv run code/infer_yolo26.py \
    --model_size l \
    --images_dir /path/to/images \
    --output_dir /custom/output \
    --weights_dir fine_tuning_weights \
    --conf 0.3 \
    --iou 0.5 \
    --font_size 20 \
    --gpu 1

# 4개 터미널 병렬 추론 예제
# Terminal 1: uv run code/infer_yolo26.py --model_size s --images_dir /data --gpu 0
# Terminal 2: uv run code/infer_yolo26.py --model_size m --images_dir /data --gpu 1  
# Terminal 3: uv run code/infer_yolo26.py --model_size l --images_dir /data --gpu 2
# Terminal 4: uv run code/infer_yolo26.py --model_size x --images_dir /data --gpu 3

# 도움말 보기
uv run code/infer_yolo26.py --help
```

### 입출력
- **입력**: 이미지 폴더 (jpg, jpeg, png, bmp 지원)
- **출력**: 
  - `predict_images/` - 바운딩 박스가 그려진 탐지 결과 이미지들
  - `predict_labels/` - YOLO 형식 라벨 파일들 (.txt)

### 출력 폴더 구조
```
inference_results/yolo26{model_size}_{timestamp}/
├── predict_images/     # 탐지 결과 이미지들
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── predict_labels/     # YOLO 형식 라벨 파일들  
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### 내부 동작 과정
1. 지정된 `weights_dir`에서 모델명을 포함하는 `.pt` 파일 자동 탐색
2. 지원 형식: `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt` 등
3. 각 이미지마다 개별 추론 수행
4. 바운딩 박스 + 라벨 + 신뢰도가 포함된 결과 이미지 저장

## 🔄 convert_json_to_yolo_ultralytics.py - 데이터 변환

COCO JSON 포맷의 어노테이션을 YOLO 형식으로 변환하는 스크립트입니다.

### 주요 기능
- **JSON → YOLO 변환**: COCO JSON을 YOLO txt 형식으로 변환
- **클래스 매핑**: 프로젝트 특화 클래스 ID 매핑 (4-8 → 0-4)
- **이미지 복사**: 훈련/검증용 이미지 자동 분류 및 복사
- **디렉토리 자동 생성**: 필요한 폴더 구조 자동 생성

### 클래스 매핑 테이블
```python
# JSON 클래스 ID → YOLO 클래스 ID 매핑
CLASS_MAPPING = {
    4: 0,  # toilet (변기)
    5: 1,  # washstand (세면대) 
    6: 2,  # sink (싱크대)
    7: 3,  # bathtub (욕조)
    8: 4   # gas_stove (가스레인지)
}
```

### 출력 구조
```
dataset/
├── data.yml            # YOLO 설정 파일
├── train/
│   ├── images/         # 훈련 이미지들
│   └── labels/         # YOLO 형식 라벨들 (.txt)
└── valid/
    ├── images/         # 검증 이미지들  
    └── labels/         # YOLO 형식 라벨들 (.txt)
```

### 사용법
```bash
# 기본 실행 (하드코딩된 경로 사용)
python code/convert_json_to_yolo_ultralytics.py

# 스크립트 내 경로 확인 필요:
# - JSON 파일 경로
# - 원본 이미지 경로  
# - 출력 dataset 경로
```

## 🏋️ yolo_train.py - 학습 유틸리티

YOLO26 모델 학습을 위한 공통 유틸리티 함수들이 포함된 모듈입니다.

### 학습 결과 저장
YOLO 모델 파인튜닝을 실행하면 `results/` 폴더에 학습 결과와 로그들이 자동으로 저장됩니다:
- **학습된 모델**: `results/{프로젝트명}/weights/best.pt` (최고 성능 모델)
- **마지막 모델**: `results/{프로젝트명}/weights/last.pt` (마지막 에포크 모델)
- **학습 로그**: `results/{프로젝트명}/results.csv` (에포크별 메트릭)
- **시각화**: `results/{프로젝트명}/` 내 confusion matrix, PR curve 등
- **설정 파일**: `results/{프로젝트명}/args.yaml` (학습 설정 저장)

### 주요 함수들
- **setup_training_environment()**: 학습 환경 초기화
- **load_model_with_config()**: 모델 및 설정 로드
- **setup_wandb()**: Weights & Biases 연동 설정
- **validate_dataset()**: 데이터셋 유효성 검사
- **save_training_results()**: 학습 결과 저장

### 사용법
```python
# 다른 스크립트에서 import하여 사용
from code.yolo_train import setup_training_environment, load_model_with_config

# 학습 환경 설정
config = setup_training_environment(model_size='s', epochs=200)

# 모델 로드
model = load_model_with_config(config)
```

## 📦 사전훈련 모델들 (yolo26*.pt)

Ultralytics에서 제공하는 YOLO26 사전훈련 모델들입니다.

### 모델 크기별 특성
- **yolo26n.pt**: 5.5MB, 가장 빠름, 정확도 낮음
- **yolo26s.pt**: 20MB, 빠름, 균형잡힌 성능  
- **yolo26m.pt**: (없음, 필요시 다운로드)
- **yolo26l.pt**: 53MB, 느림, 높은 정확도
- **yolo26x.pt**: 118MB, 가장 느림, 최고 정확도

### 용도
- **Fine-tuning 시작점**: 커스텀 데이터셋으로 전이학습시 사용
- **기준 모델**: 학습된 모델과 성능 비교용
- **빠른 테스트**: 사전훈련 성능 확인용

## 🛠️ 개발 및 디버깅 팁

### 1. 추론 스크립트 커스터마이징
```python
# infer_yolo26.py 상단의 설정값들 수정
DEFAULT_CONF = 0.3           # 더 엄격한 탐지
DEFAULT_FONT_SIZE = 24       # 큰 폰트로 라벨 표시
DEFAULT_LINE_WIDTH = 3       # 굵은 바운딩 박스
```

### 2. 새로운 이미지 형식 지원
```python
# SUPPORTED_IMAGE_EXTENSIONS에 추가
SUPPORTED_IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
```

### 3. GPU 메모리 관리
```python
# infer_yolo26.py에서 GPU 설정 변경
DEFAULT_GPU_DEVICE = '0'     # 단일 GPU
DEFAULT_GPU_DEVICE = '0,1'   # 멀티 GPU (지원시)
```

### 4. 배치 처리 최적화
현재는 이미지를 하나씩 처리하지만, 배치 처리로 변경 가능:
```python
# 여러 이미지를 한번에 처리하도록 수정
results = model.predict(source=image_list, batch_size=8)
```

## 🔧 코드 확장 가이드

### 새로운 추론 기능 추가
1. `infer_yolo26.py`의 `run_inference()` 함수 수정
2. argparse에 새로운 옵션 추가
3. 사용자 설정 변수 섹션에 기본값 정의

### 데이터 변환 스크립트 수정  
1. `convert_json_to_yolo_ultralytics.py`의 클래스 매핑 수정
2. 새로운 이미지 형식 지원 추가
3. 검증 데이터 비율 조정

### 학습 유틸리티 확장
1. `yolo_train.py`에 새로운 헬퍼 함수 추가
2. 커스텀 옵티마이저나 스케줄러 구현
3. 새로운 메트릭 계산 함수 추가

---

**참고**: 이 코드들은 메인 프로젝트의 일부이며, 전체 프로젝트 개요는 상위 디렉토리의 [README.md](../README.md)를 참조하세요.
