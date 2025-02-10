import pandas as pd
import matplotlib.pyplot as plt

# 파일 경로 지정
csv_path = "/home/a/A_2024_selfcode/NEW-PCB_Yolo/outputs_800yolo/run3/results.csv"

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# 학습 단계(에포크) 확인
epochs = df["epoch"]

# Loss 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(epochs, df["train/box_loss"], label="Train Box Loss", linestyle="--")
plt.plot(epochs, df["train/cls_loss"], label="Train Class Loss", linestyle="--")
plt.plot(epochs, df["train/dfl_loss"], label="Train DFL Loss", linestyle="--")
plt.plot(epochs, df["val/box_loss"], label="Val Box Loss")
plt.plot(epochs, df["val/cls_loss"], label="Val Class Loss")
plt.plot(epochs, df["val/dfl_loss"], label="Val DFL Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("YOLO Loss Comparison")
plt.grid()
plt.show()
