"""
YOLO26 통합 학습 스크립트 (Optimized)
- Model: YOLO26x (권장)
- Settings: Epoch 300 / Patience 50 / Batch 32
- Optimization: Built-in Early Stopping (Fitness based), Mixup Augmentation
- Fitness Score: 0.1*mAP50 + 0.9*mAP95 (Ultralytics 기본값)
"""
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


def train_yolo26(
    model_size: str,
    batch_size: int,
    epochs: int,
    patience: int,
    device: str,
    data_yaml: str,
    output_dir: str,
    use_wandb: bool,
    wandb_project: str,
    optimizer: str = 'Muon',
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    momentum: float = 0.937
):
    """
    YOLO26 모델 학습
    
    Args:
        model_size: 모델 크기 (s, m, l, x)
        batch_size: 배치 크기
        epochs: 최대 에포크 수
        patience: Early stopping patience (fitness score 기준)
        device: GPU 디바이스
        data_yaml: 데이터셋 YAML 파일 경로
        output_dir: 결과 저장 디렉토리
        use_wandb: W&B 사용 여부
        wandb_project: W&B 프로젝트명
        optimizer: 옵티마이저 ('Muon', 'Adam', 'SGD', 'AdamW' 등)
        lr: 학습률
        weight_decay: 가중치 감쇠
        momentum: 모멘텀 (SGD, Muon에만 적용)
    """
    print("="*80)
    print(f"🚀 YOLO26{model_size.upper()} 학습 시작")
    print(f"   설정: Epochs={epochs}, Patience={patience}, Batch={batch_size}")
    print(f"   디바이스: {device}")
    print(f"   기준: Fitness Score (0.1*mAP50 + 0.9*mAP95)")
    print("="*80)
    
    # 출력 디렉토리 설정
    project_name = f"yolo26{model_size}_b{batch_size}_e{epochs}"
    output_path = Path(output_dir) / project_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 모델 로드
        model_path = f'yolo26{model_size}.pt'
        print(f"\n📦 모델 로드 중: {model_path}")
        model = YOLO(model_path)
        
        print(f"\n🎯 학습 시작...")
        
        results = model.train(
            # 1. 데이터 및 기본 설정
            data=data_yaml,
            imgsz=640,
            device=device,
            
            # 2. 사용자 지정 핵심 하이퍼파라미터
            epochs=epochs,
            patience=patience,      # Fitness Score 기준 자동 중단
            batch=batch_size,
            
            # 3. 옵티마이저 설정
            optimizer=optimizer,    # 'Muon', 'Adam', 'SGD', 'AdamW' 등
            lr0=lr,                # 초기 학습률
            weight_decay=weight_decay,
            momentum=momentum,      # SGD, Muon에만 적용
            
            # 4. 저장 설정
            project=output_dir,
            name=project_name,
            save=True,              # 마지막 및 베스트 모델 저장
            save_period=10,         # 10 에폭마다 체크포인트 저장
            exist_ok=True,
            
            # 5. 성능 최적화 (YOLO26 맞춤)
            amp=True,               # Mixed Precision (속도 향상, 메모리 절약)
            cos_lr=True,            # Cosine Learning Rate Scheduler (수렴 안정성)
            
            # 6. 데이터 증강 (Large/X 모델 과적합 방지)
            mixup=0.15,             # 이미지를 섞어 학습 (X 모델 필수 추천)
            mosaic=1.0,             # 모자이크 증강 (기본값)
            
            # 7. 시스템 설정
            cache=True,             # RAM 여유 시 True (속도 향상)
            workers=8,              # Dataloader 워커 수 (CPU 코어 수에 맞게 조절)
            
            # 8. 로깅 및 시각화
            verbose=True,
            plots=True,
            
            # 9. W&B 설정 (Ultralytics 내장 기능 사용)
            project=wandb_project if use_wandb else None
        )
        
        print("\n" + "="*80)
        print("✅ 학습이 성공적으로 완료되었습니다!")
        print(f"🏆 Best Model: {output_path}/weights/best.pt")
        print("="*80)
        
        # 최종 성능 지표
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\n📊 최종 성능 지표:")
            print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
            print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
            print(f"   Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
            print(f"   Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
            fitness = 0.1 * metrics.get('metrics/mAP50(B)', 0) + 0.9 * metrics.get('metrics/mAP50-95(B)', 0)
            print(f"   Fitness Score: {fitness:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 학습 중 치명적인 오류 발생: {str(e)}")
        return False
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU 메모리 정리 완료")


def main():
    parser = argparse.ArgumentParser(description='YOLO26 통합 학습기')
    
    # 기본값 설정 (제미나이 권장 설정)
    parser.add_argument('--model_size', '-m', default='x', choices=['s', 'm', 'l', 'x'],
                        help='모델 크기 (기본값: x)')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='배치 크기 (기본값: 32)')
    parser.add_argument('--epochs', '-e', type=int, default=300,
                        help='최대 에포크 수 (기본값: 300)')
    parser.add_argument('--patience', '-p', type=int, default=50,
                        help='Early stopping patience (기본값: 50)')
    parser.add_argument('--device', '-d', default='0,1,2,3',
                        help='GPU 디바이스 (기본값: 0,1,2,3)')
    
    # 옵티마이저 설정
    parser.add_argument('--optimizer', '-o', default='Muon',
                        choices=['Muon', 'Adam', 'AdamW', 'SGD', 'RMSprop'],
                        help='옵티마이저 (기본값: Muon)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='학습률 (기본값: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='가중치 감쇠 (기본값: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='모멘텀 (기본값: 0.937)')
    
    # 경로 설정
    parser.add_argument('--data_yaml', default='/home/themiraclesoft/wishket/dataset/data.yml',
                        help='데이터셋 YAML 파일 경로')
    parser.add_argument('--output_dir', default='/home/themiraclesoft/wishket/results',
                        help='결과 저장 디렉토리')
    
    # W&B 설정
    parser.add_argument('--use_wandb', action='store_true',
                        help='Weights & Biases 사용')
    parser.add_argument('--wandb_project', default='yolo26-construction',
                        help='W&B 프로젝트명')
    
    args = parser.parse_args()
    
    # 학습 실행
    success = train_yolo26(
        model_size=args.model_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum
    )
    
    if success:
        print("\n🎉 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n⚠️ 작업이 실패했습니다. 로그를 확인하세요.")
        exit(1)


if __name__ == "__main__":
    main()
