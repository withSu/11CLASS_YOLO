from ultralytics import YOLO

# 모델 로드
model = YOLO('/home/a/A_2024_selfcode/PCB/scripts/runs/obb/train24/weights/best.pt')


# 모델 추론 실행 
results = model.predict(
    source='/home/a/A_2024_selfcode/PCB/dataset/test/images',
    save=True,        # 이미지만 저장
    save_txt=True,    # 라벨 텍스트 파일도 저장
    conf=0.1
)


