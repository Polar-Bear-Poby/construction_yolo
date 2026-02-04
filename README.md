# YOLO26 건축 도면 객체 탐지 프로젝트

YOLO26 모델을 사용한 건축 도면 내 화장실 관련 객체 탐지 시스템입니다.

## 🚀 빠른 시작

```bash
# 전체 모델 단계별 학습
./run_all_training_cli.sh

# 개별 모델 학습
python train_yolo26_cli.py --model_size s --epochs 200
```

## 📁 주요 파일

- `run_all_training_cli.sh` - 메인 단계별 학습 스크립트
- `train_yolo26_cli.py` - 개별 모델 학습 스크립트  
- `code/convert_json_to_yolo_ultralytics.py` - JSON→YOLO 변환
- `code/yolo_train.py` - 학습 유틸리티

자세한 사용법은 [code/README.md](code/README.md)를 참조하세요.
