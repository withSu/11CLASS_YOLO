import os
import cv2
import numpy as np

def visualize_labels(label_dir, image_dir, output_dir, is_obb=True, alpha=0.5):
    # output 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 모든 라벨 파일 처리
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            base_name = os.path.splitext(label_file)[0]
            label_path = os.path.join(label_dir, label_file)

            # 이미지 파일 경로 검색
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

            # 복사본 생성 (투명도 적용용)
            overlay = image.copy()

            # 라벨 파일 읽기
            with open(label_path, 'r') as f:
                labels = f.readlines()

            # 라벨에 대한 색상 매핑 (랜덤 색상)
            label_colors = {}

            # 라벨 데이터 시각화
            for label in labels:
                parts = label.strip().split()
                class_id = int(parts[0])

                # 클래스에 색상이 없으면 생성
                if class_id not in label_colors:
                    label_colors[class_id] = tuple(np.random.randint(0, 255, 3).tolist())

                if is_obb:
                    # OBB (Oriented Bounding Box)
                    points = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
                    points[:, 0] *= image.shape[1]  # 가로 방향 정규화 해제
                    points[:, 1] *= image.shape[0]  # 세로 방향 정규화 해제
                    points = points.astype(np.int32)
                    cv2.fillPoly(overlay, [points], color=label_colors[class_id])  # 영역을 색칠
                    x, y = points[0]  # 첫 번째 꼭짓점 기준으로 텍스트 위치

                    # 면적 계산
                    x_coords = points[:, 0]
                    y_coords = points[:, 1]
                    area_px = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                else:
                    # YOLO 형식 (x_center, y_center, width, height)
                    x_center, y_center, width, height = map(float, parts[1:])
                    x_center *= image.shape[1]
                    y_center *= image.shape[0]
                    width *= image.shape[1]
                    height *= image.shape[0]

                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    rect_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    cv2.fillPoly(overlay, [rect_points], color=label_colors[class_id])  # 영역을 색칠
                    x, y = x1, y1  # 좌상단 기준으로 텍스트 위치

                    # 면적 계산
                    area_px = (x2 - x1) * (y2 - y1)

                # 클래스 ID와 픽셀 값 텍스트 추가
                cv2.putText(
                    image, f"Class {class_id}", (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_colors[class_id], 2
                )
                cv2.putText(
                    image, f"{int(area_px)} px²", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_colors[class_id], 2
                )

            # 투명도 적용
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # 시각화된 이미지를 저장
            output_path = os.path.join(output_dir, f"{base_name}_visualized.jpg")
            cv2.imwrite(output_path, image)
            print(f"시각화된 이미지 저장 완료: {output_path}")

if __name__ == "__main__":
    visualize_labels(
        
        label_dir="/home/a/A_2024_selfcode/PCB_yolo/scripts/runs/obb/predict4/labels",   # 추론 라벨 경로
        image_dir="/home/a/A_2024_selfcode/PCB_yolo/dataset/test/images",  # 추론 이미지 경로
        output_dir="/home/a/A_2024_selfcode/PCB_yolo/scripts/runs/obb/predict4/visualized",  # 저장 경로
        is_obb=True,  # OBB 형식(True) 또는 YOLO 형식(False) 선택
        alpha=0.5  # 투명도 (0.0: 완전 투명, 1.0: 불투명)
    )




