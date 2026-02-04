"""
JSON 라벨을 YOLO TXT 형식으로 변환하는 스크립트
- Ultralytics의 convert_coco 함수를 사용하여 변환
- 이미지별 JSON 파일들을 하나의 표준 COCO JSON으로 합친 후 변환
- train과 valid 모두 처리
"""

import json
import os
from pathlib import Path
from ultralytics.data.converter import convert_coco

def merge_json_to_coco(labels_dir: str, output_json: str):
    """
    이미지별 JSON 파일들을 하나의 표준 COCO JSON으로 합침

    Args:
        labels_dir: JSON 파일들이 있는 디렉토리
        output_json: 합쳐진 COCO JSON 파일 경로
    """
    labels_path = Path(labels_dir)
    json_files = list(labels_path.glob('*.json'))

    # COCO 형식 초기화 (관심 클래스만 포함하여 Ultralytics convert_coco가 자동으로 0부터 재매핑하도록)
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "toilet"},
            {"id": 2, "name": "washstand"},
            {"id": 3, "name": "sink"},
            {"id": 4, "name": "bathtub"},
            {"id": 5, "name": "gas_stove"}
        ]
    }

    image_id_counter = 1
    annotation_id_counter = 0

    print(f"총 {len(json_files)}개 JSON 파일을 COCO 형식으로 합치는 중...")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 이미지 정보 추가
            if data['images']:
                image = data['images'][0]
                image_entry = {
                    "id": image_id_counter,
                    "width": image['width'],
                    "height": image['height'],
                    "file_name": image['file_name']
                }
                coco_data["images"].append(image_entry)

                # 어노테이션 추가 (관심 클래스만 필터링, category_id를 1부터 시작하도록 매핑하여 Ultralytics가 cls = cat_id - 1로 0부터 매핑)
                # 원본: 4=toilet, 5=washstand, 6=sink, 7=bathtub, 8=gas_stove
                # 매핑: 4->1, 5->2, 6->3, 7->4, 8->5
                # Ultralytics convert_coco: cls = cat_id - 1 -> 0,1,2,3,4
                class_mapping = {4: 1, 5: 2, 6: 3, 7: 4, 8: 5}

                for ann in data['annotations']:
                    category_id = ann['category_id']
                    if category_id in class_mapping:
                        mapped_id = class_mapping[category_id]
                        annotation_entry = {
                            "id": annotation_id_counter,
                            "image_id": image_id_counter,
                            "category_id": mapped_id,  # 매핑된 ID 사용
                            "bbox": ann['bbox'],
                            "area": ann['area'],
                            "iscrowd": ann['iscrowd']
                        }
                        coco_data["annotations"].append(annotation_entry)
                        annotation_id_counter += 1

                image_id_counter += 1
                print(f"✅ 처리 완료: {json_file.name}")

        except Exception as e:
            print(f"❌ 처리 실패: {json_file.name} - {str(e)}")

    # COCO JSON 저장
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)

    print(f"🎉 COCO JSON 생성 완료: {output_json}")
    return output_json

def convert_using_ultralytics(labels_dir: str, save_dir: str = None):
    """
    Ultralytics의 convert_coco 함수를 사용하여 변환

    Args:
        labels_dir: JSON 파일들이 있는 디렉토리
        save_dir: YOLO TXT 저장 디렉토리
    """
    # 임시 COCO JSON 생성
    temp_coco_json = os.path.join(labels_dir, "annotations.json")
    merge_json_to_coco(labels_dir, temp_coco_json)

    # labels 디렉토리 설정
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(labels_dir), "labels")
    
    # labels 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    print("Ultralytics convert_coco 실행 중...")
    convert_coco(
        labels_dir=labels_dir,  # annotations.json이 있는 폴더
        save_dir=save_dir,      # YOLO TXT 저장 폴더
        use_segments=False,     # 박스 학습
        cls91to80=False         # COCO 80클래스 표준 아님
    )

    # 임시 파일 정리
    if os.path.exists(temp_coco_json):
        os.remove(temp_coco_json)
        print("임시 파일 정리 완료")

def convert_all_labels():
    """
    train과 valid labels 모두 변환 (labels_backup에서 변환하여 dataset에 저장)
    """
    base_path = "/home/themiraclesoft/wishket"

    # train labels 변환 (labels_backup -> dataset)
    train_labels_dir = os.path.join(base_path, "labels_backup", "train", "labels")
    train_save_dir = os.path.join(base_path, "dataset", "train")
    if os.path.exists(train_labels_dir):
        print(f"🔄 Train labels 변환 시작: {train_labels_dir} -> {train_save_dir}")
        convert_using_ultralytics(train_labels_dir, train_save_dir)
        print("✅ Train labels 변환 완료\n")
    else:
        print(f"❌ Train labels 디렉토리가 존재하지 않음: {train_labels_dir}\n")

    # valid labels 변환 (labels_backup -> dataset)
    valid_labels_dir = os.path.join(base_path, "labels_backup", "valid", "labels")
    valid_save_dir = os.path.join(base_path, "dataset", "valid")
    if os.path.exists(valid_labels_dir):
        print(f"🔄 Valid labels 변환 시작: {valid_labels_dir} -> {valid_save_dir}")
        convert_using_ultralytics(valid_labels_dir, valid_save_dir)
        print("✅ Valid labels 변환 완료\n")
    else:
        print(f"❌ Valid labels 디렉토리가 존재하지 않음: {valid_labels_dir}\n")

if __name__ == "__main__":
    convert_all_labels()