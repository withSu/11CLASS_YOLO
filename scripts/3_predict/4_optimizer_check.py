import torch

# 체크포인트 파일 경로
checkpoint_path = "/home/a/A_2024_selfcode/PCB_yolo/scripts/runs/obb/train25/weights/last.pt"

# 체크포인트 로드
checkpoint = torch.load(checkpoint_path)

# Optimizer 상태 확인
if "optimizer" in checkpoint:
    optimizer_state = checkpoint["optimizer"]
    print(f"Optimizer type: {type(optimizer_state).__name__}")
else:
    print("Optimizer information not found in checkpoint.")
