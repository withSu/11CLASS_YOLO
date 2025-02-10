# input: 3_new_raw_json / 1_images
# output: 1_2_800images, 4_800labels
import os
import json
from PIL import Image

# 원본 해상도
ORIGINAL_WIDTH = 3904
ORIGINAL_HEIGHT = 3904

# 목표 해상도
TARGET_WIDTH = 800
TARGET_HEIGHT = 800

# 클래스 매핑 (YOLO 형식은 숫자 클래스 ID를 사용함)
CLASS_NAMES = {
    'component': 0,
}

# 디렉토리 설정
JSON_INPUT_DIR = '/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset2/3_new_raw_json'
IMAGE_INPUT_DIR = '/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset2/1_images'
IMAGE_OUTPUT_DIR = '/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset2/1_2_800images'
LABEL_OUTPUT_DIR = '/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset2/4_800labels'

# 출력 디렉토리 생성
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(LABEL_OUTPUT_DIR, exist_ok=True)

def convert_and_resize(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JSON 파일명으로부터 이미지명 추론
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    # 확장자는 사용자가 jpg, png 등 다양할 수 있으므로 뒤에서 찾는다
    image_file = None
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = os.path.join(IMAGE_INPUT_DIR, base_name + ext)
        if os.path.exists(candidate):
            image_file = candidate
            break
    
    if not image_file:
        print(f"⚠️ 이미지가 없습니다: {base_name}")
        return
    
    # 이미지 로드 후 리사이즈
    try:
        img = Image.open(image_file)
        img_resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
    except Exception as e:
        print(f"❌ 이미지 리사이즈 실패: {image_file}\n{e}")
        return
    
    # 리사이즈된 이미지 저장 (이름은 원본과 동일, 경로만 변경)
    resized_image_path = os.path.join(IMAGE_OUTPUT_DIR, os.path.basename(image_file))
    img_resized.save(resized_image_path)
    
    # 라벨 파일로 저장할 경로
    label_path = os.path.join(LABEL_OUTPUT_DIR, base_name + ".txt")
    
    # JSON의 원본 크기가 3904x3904라고 가정해 [0..1] 범위로 정규화
    image_width = data.get('imageWidth', ORIGINAL_WIDTH)
    image_height = data.get('imageHeight', ORIGINAL_HEIGHT)
    
    # 라벨 파일 생성
    with open(label_path, 'w', encoding='utf-8') as out_f:
        for shape in data.get('shapes', []):
            label = shape.get('label', 'unknown')
            points = shape.get('points', [])

            if label not in CLASS_NAMES:
                print(f"⚠️ 알 수 없는 라벨: '{label}' → {json_path}, 스킵")
                continue
            class_id = CLASS_NAMES[label]

            # 좌표 정규화 (0..1)
            if len(points) == 4:  # 꼭짓점 4개 → OBB
                x_coords = [p[0] / image_width for p in points]
                y_coords = [p[1] / image_height for p in points]
                out_f.write(
                    f"{class_id} " +
                    " ".join([f"{x:.6f} {y:.6f}" for x, y in zip(x_coords, y_coords)]) + "\n"
                )
            elif len(points) == 2:  # 두 점만 있는 경우 → 사각형
                x1, y1 = points[0]
                x2, y2 = points[1]
                rect_points = [
                    [x1 / image_width, y1 / image_height],  # 좌상단
                    [x2 / image_width, y1 / image_height],  # 우상단
                    [x2 / image_width, y2 / image_height],  # 우하단
                    [x1 / image_width, y2 / image_height],  # 좌하단
                ]
                out_f.write(
                    f"{class_id} " +
                    " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in rect_points]) + "\n"
                )
            else:
                print(f"❌ 지원하지 않는 도형 (points = {len(points)}개): {json_path}, 스킵")
                continue
    
    print(f"✅ 변환 완료: {json_path} → {label_path}, 이미지 리사이즈 완료")

def main():
    # 모든 JSON 파일 순회
    json_files = [f for f in os.listdir(JSON_INPUT_DIR) if f.endswith(".json")]
    if not json_files:
        print("❗ JSON 파일이 없습니다.")
        return
    
    for json_file in json_files:
        json_path = os.path.join(JSON_INPUT_DIR, json_file)
        convert_and_resize(json_path)

    print("✅ 모든 변환 및 리사이즈를 완료했습니다.")

if __name__ == "__main__":
    main()
