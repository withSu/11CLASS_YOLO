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

def visualize_labels(label_dir, image_dir, output_dir):
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
                    color = (0, 0, 255)  # 빨간색
                elif size_label == "Medium":
                    color = (255, 0, 0)  # 파란색
                else:
                    color = (0, 255, 255)  # 노란색

                # 테두리 사각형
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # 클래스 이름만 표시 (두께 줄임)
                cv2.putText(
                    image, class_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

            # 결과 저장
            output_path = os.path.join(output_dir, f"{base_name}_visualized.jpg")
            cv2.imwrite(output_path, image)
            print(f"시각화된 이미지 저장 완료: {output_path}")

if __name__ == "__main__":
    visualize_labels(
        label_dir="/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/4_800size_txt_labels",  # txt 라벨 디렉토리
        image_dir="/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/1_1_800images",  # 이미지 디렉토리
        output_dir="/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/5_lets_visualize_coco"  # 결과 저장 디렉토리
    )
