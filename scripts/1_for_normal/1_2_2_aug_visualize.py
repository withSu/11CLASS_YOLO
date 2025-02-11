import os
import random
import cv2
import matplotlib.pyplot as plt
from glob import glob

print("바운딩 박스 시각화 코드를 실행한다.")

# 증강된 이미지와 라벨이 저장된 폴더 경로를 지정한다.
augmented_dir = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/5_1_yolo_augmented_output"
# 시각화 결과를 저장할 폴더가 필요할 경우 생성한다.
visualized_output_dir = "/home/a/A_2024_selfcode/CLASS-PCB_Yolo/dataset/5_2_yolo_augmented_visualized"
if not os.path.exists(visualized_output_dir):
    os.makedirs(visualized_output_dir)
    print(f"생성: {visualized_output_dir}")

# augmented_dir 내의 모든 jpg 이미지 파일 목록을 가져온다.
image_files = glob(os.path.join(augmented_dir, "*.jpg"))
if len(image_files) < 50:
    print(f"주의: 50장보다 적은 이미지({len(image_files)}개)가 발견되었다.")
    sample_images = image_files
else:
    sample_images = random.sample(image_files, 50)

# matplotlib의 서브플롯을 사용하여 5행 10열의 그리드로 이미지를 출력한다.
fig, axes = plt.subplots(5, 10, figsize=(20, 10))
axes = axes.flatten()

for ax, img_path in zip(axes, sample_images):
    # 이미지를 OpenCV로 읽고 BGR에서 RGB로 변환한다.
    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지 파일명에서 확장자를 제거하여 라벨 파일명을 구성한다.
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(augmented_dir, base_name + ".txt")
    
    # 라벨 파일이 존재하면 YOLO 형식의 바운딩 박스 정보를 읽는다.
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
        img_h, img_w, _ = image.shape
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            # 첫 번째 값은 클래스 번호, 이후 값들은 x_center, y_center, width, height이다.
            cls = parts[0]
            x_center, y_center, w, h = map(float, parts[1:])
            # YOLO 좌표를 픽셀 좌표로 변환한다.
            x_min = int((x_center - w / 2) * img_w)
            y_min = int((y_center - h / 2) * img_h)
            x_max = int((x_center + w / 2) * img_w)
            y_max = int((y_center + h / 2) * img_h)
            # 바운딩 박스를 그린다.
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(image, str(cls), (x_min, max(y_min - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)
    ax.imshow(image)
    ax.axis("off")

plt.tight_layout()
plt.show()
