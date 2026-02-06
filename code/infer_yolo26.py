"""
YOLO26 학습된 모델 추론 스크립트
- 학습된 best.pt 모델들을 사용하여 validation 데이터 추론
- 추론 결과 이미지 저장 및 메트릭 계산
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
from dotenv import load_dotenv

# 프로젝트 루트 자동 감지
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# ===== 사용자 설정 변수들 =====
# 학습된 모델들이 저장된 디렉토리
DEFAULT_WEIGHTS_DIR = 'fine_tuning_weights'  # 기본 가중치 폴더

# 기본 추론 설정
DEFAULT_CONF = 0.25          # Confidence threshold
DEFAULT_IOU = 0.45           # IoU threshold
DEFAULT_FONT_SIZE = 16       # 라벨 폰트 크기
DEFAULT_LINE_WIDTH = 2       # 바운딩 박스 선 두께
DEFAULT_GPU_DEVICE = '0'     # 사용할 GPU 디바이스

# 지원하는 이미지 확장자
SUPPORTED_IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

# 기본 출력 디렉토리 패턴 (상대 경로)
DEFAULT_OUTPUT_PATTERN = "inference_results/yolo26{model_size}_{timestamp}"
# ================================

# 환경 변수 로드 (프로젝트 루트 기준)
load_dotenv(PROJECT_ROOT / '.env')


def run_inference(model_path: str, images_dir: str, output_dir: str, conf: float = DEFAULT_CONF, 
                 iou: float = DEFAULT_IOU, font_size: int = DEFAULT_FONT_SIZE, gpu_device: str = DEFAULT_GPU_DEVICE):
    """
    단일 모델로 추론 수행

    Args:
        model_path: 학습된 모델 경로 (best.pt)
        images_dir: 추론할 이미지들이 있는 디렉토리
        output_dir: 결과 저장 디렉토리
        conf: confidence threshold
        iou: IoU threshold
        font_size: 라벨 폰트 크기 (커스텀 시각화로 구현)
        gpu_device: 사용할 GPU 디바이스 번호
    """
    print(f"\n🚀 모델 추론 시작: {Path(model_path).name}")
    print(f"   설정: conf={conf}, iou={iou}, font_size={font_size}, gpu={gpu_device}")
    print(f"   출력: {output_dir}")

    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # 모델 로드
        print(f"\n📦 모델 로드 중: {model_path}")
        model = YOLO(model_path)

        # 이미지 디렉토리 확인
        if not images_dir or not os.path.exists(images_dir):
            print(f"❌ 이미지 디렉토리를 찾을 수 없음: {images_dir}")
            return False

        print(f"   이미지 데이터: {images_dir}")

        # 이미지 파일 목록 가져오기
        import glob
        image_files = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
            image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        
        if not image_files:
            print(f"❌ 이미지 파일을 찾을 수 없음: {images_dir}")
            return False

        print(f"   총 이미지 수: {len(image_files)}")
        
        # predict_images와 predict_labels 디렉토리 생성
        predict_images_dir = output_path / "predict_images"
        predict_labels_dir = output_path / "predict_labels"
        predict_images_dir.mkdir(parents=True, exist_ok=True)
        predict_labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎯 이미지 1장씩 추론 시작... (총 {len(image_files)}장)")
        
        # 이미지 1장씩 추론
        for i, image_path in enumerate(image_files, 1):
            image_name = Path(image_path).stem
            print(f"   [{i:4d}/{len(image_files)}] {image_name}")
            
            # 개별 이미지 추론 (시각화는 직접 처리)
            results = model.predict(
                source=image_path,
                conf=conf,
                iou=iou,
                save=False,  # 자동 저장 비활성화
                save_txt=False,  # 텍스트 저장도 비활성화 (수동으로 처리)
                verbose=False,  # 개별 이미지마다 로그 출력 방지
                device=gpu_device,
                augment=False
            )
            
            # 결과 처리
            if len(results) > 0:
                result = results[0]
                
                # 원본 이미지 로드
                import cv2
                import numpy as np
                
                orig_img = cv2.imread(image_path)
                img_height, img_width = orig_img.shape[:2]
                
                # 라벨 파일 내용 (YOLO 형식)
                label_lines = []
                
                # 탐지된 객체들 처리
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # 박스 정보 추출
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # YOLO 형식 좌표 변환 (정규화)
                        center_x = (x1 + x2) / 2 / img_width
                        center_y = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        # 라벨 파일 라인 추가
                        label_lines.append(f"{cls_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {confidence:.6f}")
                        
                        # 바운딩 박스 그리기
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), DEFAULT_LINE_WIDTH)
                        
                        # 클래스 이름 매핑 (프로젝트 특화)
                        class_names = {0: 'toilet', 1: 'washstand', 2: 'sink', 3: 'bathtub', 4: 'gas_stove'}
                        class_name = class_names.get(cls_id, f'class{cls_id}')
                        
                        # 라벨 텍스트 (커스텀 폰트 크기 적용)
                        label = f"{class_name}: {confidence:.2f}"
                        font_scale = font_size / 16.0  # 기본값 16 기준으로 스케일링
                        
                        # 텍스트 크기 계산
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                        
                        # 텍스트 배경 박스
                        cv2.rectangle(orig_img, (x1, y1 - text_height - baseline - 5),
                                    (x1 + text_width, y1), (0, 255, 0), -1)
                        
                        # 텍스트 그리기 (검은색)
                        cv2.putText(orig_img, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
                
                # 결과 이미지 저장
                output_img_path = predict_images_dir / f"{image_name}.jpg"
                cv2.imwrite(str(output_img_path), orig_img)
                
                # 라벨 파일 저장
                if label_lines:
                    output_label_path = predict_labels_dir / f"{image_name}.txt"
                    with open(output_label_path, 'w') as f:
                        f.write('\n'.join(label_lines))
                else:
                    # 탐지된 객체가 없어도 빈 라벨 파일 생성
                    output_label_path = predict_labels_dir / f"{image_name}.txt"
                    with open(output_label_path, 'w') as f:
                        pass  # 빈 파일

        print(f"\n✅ 이미지별 추론 완료!")
        print(f"   이미지 결과: {predict_images_dir}")
        print(f"   라벨 결과: {predict_labels_dir}")
        print(f"   처리된 이미지: {len(image_files)}장")

        return True

    except Exception as e:
        print(f"\n❌ 추론 중 오류 발생: {str(e)}")
        return False

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='YOLO26 학습된 모델 추론기')

    # 모델 선택
    parser.add_argument('--model_size', '-m', choices=['s', 'm', 'l', 'x'],
                        default='s', help='추론할 모델 크기')

    # 입력 설정
    parser.add_argument('--images_dir', '-i', required=True,
                        help='추론할 이미지들이 있는 디렉토리')
    parser.add_argument('--weights_dir', '-w', default=DEFAULT_WEIGHTS_DIR,
                        help=f'가중치 파일들이 있는 디렉토리 (기본값: {DEFAULT_WEIGHTS_DIR})')
    parser.add_argument('--output_dir', '-o', default=None,
                        help='추론 결과 저장 디렉토리 (미지정 시 자동 생성)')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF,
                        help=f'Confidence threshold (기본값: {DEFAULT_CONF})')
    parser.add_argument('--iou', type=float, default=DEFAULT_IOU,
                        help=f'IoU threshold (기본값: {DEFAULT_IOU})')
    parser.add_argument('--font_size', type=int, default=DEFAULT_FONT_SIZE,
                        help=f'라벨 폰트 크기 (기본값: {DEFAULT_FONT_SIZE})')
    parser.add_argument('--gpu', type=str, default=DEFAULT_GPU_DEVICE,
                        help=f'사용할 GPU 디바이스 번호 (기본값: {DEFAULT_GPU_DEVICE})')

    args = parser.parse_args()

    print("🎯 YOLO26 학습된 모델 추론 시작")
    print("="*60)

    # 지정된 가중치 폴더에서 모델 파일들 자동 찾기
    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        print(f"❌ 가중치 폴더를 찾을 수 없습니다: {weights_dir}")
        return

    model_paths = {}
    
    # 가중치 폴더에서 모든 .pt 파일 찾기
    for pt_file in weights_dir.glob('*.pt'):
        if pt_file.is_file():
            filename = pt_file.name.lower()
            # 모델 크기 추출 (파일명에 모델명이 포함된 경우)
            if 'yolo26s' in filename or 'yolov8s' in filename:
                model_paths['s'] = str(pt_file)
            elif 'yolo26m' in filename or 'yolov8m' in filename:
                model_paths['m'] = str(pt_file)
            elif 'yolo26l' in filename or 'yolov8l' in filename:
                model_paths['l'] = str(pt_file)
            elif 'yolo26x' in filename or 'yolov8x' in filename:
                model_paths['x'] = str(pt_file)

    if not model_paths:
        print("❌ 가중치 폴더에서 모델 파일(.pt)을 찾을 수 없습니다!")
        print(f"   확인 경로: {weights_dir}")
        print("   지원 형식: yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt")
        return

    print(f"✅ 발견된 모델들 (경로: {weights_dir}):")
    for size, path in model_paths.items():
        print(f"   YOLO26{size.upper()}: {Path(path).name}")

    # 선택된 모델 확인
    if args.model_size not in model_paths:
        print(f"❌ YOLO26{args.model_size.upper()} 모델을 찾을 수 없습니다!")
        return

    model_path = model_paths[args.model_size]
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없음: {model_path}")
        return

    # 출력 디렉토리 설정 (자동 생성)
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = DEFAULT_OUTPUT_PATTERN.format(
            model_size=args.model_size, 
            timestamp=timestamp
        )
        print(f"📁 자동 생성된 출력 폴더: {args.output_dir}")

    # 이미지 디렉토리 확인
    if not os.path.exists(args.images_dir):
        print(f"❌ 이미지 디렉토리를 찾을 수 없음: {args.images_dir}")
        return

    print(f"\n🎯 선택된 모델: YOLO26{args.model_size.upper()}")
    print(f"📂 이미지 경로: {args.images_dir}")

    # 단일 모델로 추론 실행
    success = run_inference(
        model_path=model_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        font_size=args.font_size,
        gpu_device=args.gpu
    )

    print("\n" + "="*60)
    if success:
        print(f"🎉 추론 작업 완료!")
        print(f"📁 결과 확인: {args.output_dir}")
    else:
        print(f"❌ 추론 작업 실패!")
    print("="*60)


if __name__ == "__main__":
    main()