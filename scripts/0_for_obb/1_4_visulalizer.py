# input: 4_800labels, 1_2_800images
# output: 6_lets_visualize_coco
# 색상별 바운딩 박스 시각화
import os
import cv2
import numpy as np

# 예시 클래스 매핑 (추가 가능)

CLASS_NAMES = {
    0: "component"
}

def get_coco_size_label(w, h):
    """COCO 기준 (면적 기반)으로 Small/Medium/Large 분류"""
    area = w * h
    if area < 32**2:  # 1024 미만
        return "Small"
    elif 32**2 <= area < 96**2:  # 1024 이상, 9216 미만
        return "Medium"
    else:  # 9216 이상
        return "Large"

def visualize_labels(label_dir, image_dir, output_dir, is_obb=True):
    # output 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            base_name = os.path.splitext(label_file)[0]
            label_path = os.path.join(label_dir, label_file)

            # 이미지 파일 검색
            for ext in ['.jpg', '.png', '.jpeg']:
                image_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(image_path):
                    break
            else:
                print(f"이미지가 없습니다: {base_name}")
                continue

            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                print(f"이미지를 불러올 수 없습니다: {image_path}")
                continue

            # 라벨 파일 읽기
            with open(label_path, 'r') as f:
                labels = f.readlines()

            for label in labels:
                parts = label.strip().split()
                class_id = int(parts[0])
                class_name = CLASS_NAMES.get(class_id, f"cls_{class_id}")

                if is_obb:
                    # OBB (Oriented Bounding Box)
                    # parts[1:] = x1 y1 x2 y2 x3 y3 x4 y4 (정규화)
                    points = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
                    # 이미지 크기에 맞게 복원
                    points[:, 0] *= image.shape[1]
                    points[:, 1] *= image.shape[0]
                    points = points.astype(int)

                    x_min, y_min = points[:,0].min(), points[:,1].min()
                    x_max, y_max = points[:,0].max(), points[:,1].max()
                    w = x_max - x_min
                    h = y_max - y_min

                    # COCO 기준 크기분류
                    size_label = get_coco_size_label(w, h)
                    # 테두리 색상 (Small=빨강, Medium=파랑, Large=노랑)
                    if size_label == "Small":
                        color = (0, 0, 255)     # BGR (빨강)
                    elif size_label == "Medium":
                        color = (255, 0, 0)    # 파랑
                    else:
                        color = (0, 255, 255)  # 노랑

                    # 바운딩 박스 테두리 그리기
                    cv2.polylines(image, [points], True, color, 2)

                    # 정보 표시 (클래스, w×h, 면적)
                    area_px = w * h
                    text = f"{class_name} {w}x{h} : {area_px}"
                    x_text, y_text = points[0][0], points[0][1] - 5
                    # 너무 위면 아래로 표시
                    if y_text < 10:
                        y_text = points[0][1] + 15

                    # 글씨 크기 줄이고(0.4), 두께도 줄이기(1)
                    cv2.putText(
                        image, text, (x_text, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                    )

                else:
                    # YOLO (x_center, y_center, width, height) 정규화
                    x_center, y_center, w, h = map(float, parts[1:])
                    iw, ih = image.shape[1], image.shape[0]
                    x_center *= iw
                    y_center *= ih
                    w *= iw
                    h *= ih

                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)

                    size_label = get_coco_size_label(w, h)
                    if size_label == "Small":
                        color = (0, 0, 255)
                    elif size_label == "Medium":
                        color = (255, 0, 0)
                    else:
                        color = (0, 255, 255)

                    # 테두리 사각형
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    # 정보 표시
                    area_px = int(w * h)
                    text = f"{class_name} {int(w)}x{int(h)} : {area_px}"
                    cv2.putText(
                        image, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                    )

            # 결과 저장
            output_path = os.path.join(output_dir, f"{base_name}_visualized.jpg")
            cv2.imwrite(output_path, image)
            print(f"시각화된 이미지 저장 완료: {output_path}")


if __name__ == "__main__":
    visualize_labels(
        label_dir="/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset2/4_800labels",  # txt 라벨 디렉토리
        image_dir="/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset2/1_2_800images",  # 이미지 디렉토리

        output_dir="/home/a/A_2024_selfcode/NEW-PCB_Yolo/dataset2/6_lets_visualize_coco",  # 결과 저장 디렉토리
        is_obb=True
    )

