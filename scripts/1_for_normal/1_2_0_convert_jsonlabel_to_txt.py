# convert_labels.py

import os
import json

# 원본 해상도 (원본은 3904, 변환된 해상도는 800)
ORIGINAL_WIDTH = 3904
ORIGINAL_HEIGHT = 3904
TARGET_WIDTH = 800
TARGET_HEIGHT = 800

# 클래스 매핑 (YOLO 형식은 숫자 클래스 ID를 사용함)
CLASS_NAMES = {
    'Chip': 0,
    'Solder': 1,
    '2sideIC': 2,
    'SOD': 3,
    'Circle': 4,
    '4sideIC': 5,
    'Tantalum': 6,
    'BGA': 7,
    'MELF': 8,
    'Crystal': 9,
    'Array': 10
}

# 디렉토리 설정
JSON_INPUT_DIR = '/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/3_new_raw_json_labels'
LABEL_OUTPUT_DIR = '/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/4_800size_txt_labels'

# 출력 디렉토리 생성
os.makedirs(LABEL_OUTPUT_DIR, exist_ok=True)

def convert_labels(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JSON 파일명으로부터 라벨명 추론
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    label_path = os.path.join(LABEL_OUTPUT_DIR, base_name + ".txt")
    
    # 라벨 파일 생성 (YOLO 형식 적용)
    with open(label_path, 'w', encoding='utf-8') as out_f:
        for shape in data.get('shapes', []):
            label = shape.get('label', 'unknown')
            points = shape.get('points', [])

            # 라벨이 CLASS_NAMES에 없으면 스킵
            if label not in CLASS_NAMES:
                print(f"⚠️ 알 수 없는 라벨: '{label}' → {json_path}, 스킵")
                continue

            class_id = CLASS_NAMES[label]

            # points 개수에 따른 처리
            if len(points) == 2:
                # 꼭짓점이 2개인 경우: [ (x1, y1), (x2, y2) ]
                x1, y1 = points[0]
                x2, y2 = points[1]
            elif len(points) == 4:
                # 꼭짓점이 4개인 경우: AABB 형태로 변환
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
            else:
                print(f"❌ 지원하지 않는 도형 (points = {len(points)}개): {json_path}, 스킵")
                continue

            # 해상도 조정 (3904 -> 800 기준 변환)
            x1 = (x1 / ORIGINAL_WIDTH) * TARGET_WIDTH
            y1 = (y1 / ORIGINAL_HEIGHT) * TARGET_HEIGHT
            x2 = (x2 / ORIGINAL_WIDTH) * TARGET_WIDTH
            y2 = (y2 / ORIGINAL_HEIGHT) * TARGET_HEIGHT
            
            # 중심 좌표 및 너비/높이 계산 (YOLO 형식)
            x_center = ((x1 + x2) / 2) / TARGET_WIDTH
            y_center = ((y1 + y2) / 2) / TARGET_HEIGHT
            width = abs(x2 - x1) / TARGET_WIDTH
            height = abs(y2 - y1) / TARGET_HEIGHT
            
            out_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"✅ 변환 완료: {json_path} → {label_path}")

def main():
    json_files = [f for f in os.listdir(JSON_INPUT_DIR) if f.endswith(".json")]
    if not json_files:
        print("❗ JSON 파일이 없습니다.")
        return
    
    for json_file in json_files:
        json_path = os.path.join(JSON_INPUT_DIR, json_file)
        convert_labels(json_path)

    print("✅ 모든 라벨 변환을 완료했습니다.")

if __name__ == "__main__":
    main()
