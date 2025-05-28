import json
import matplotlib.pyplot as plt

# trainer_state.json 경로
state_path = "/home/elicer/jny/result/segformer-b0/checkpoint-2300/trainer_state.json"

# JSON 로드
with open(state_path, "r") as f:
    state = json.load(f)

log_history = state["log_history"]
eval_logs = [log for log in log_history if "eval_loss" in log]

# 에폭과 클래스별 IoU 추출
epochs = [log["epoch"] for log in eval_logs]
driving_iou = [log.get("eval_Driving Area_IoU", 0) for log in eval_logs]
parking_iou = [log.get("eval_Parking Area_IoU", 0) for log in eval_logs]
no_parking_iou = [log.get("eval_No Parking Area_IoU", 0) for log in eval_logs]

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(epochs, driving_iou, marker="o", label="Driving Area")
plt.plot(epochs, parking_iou, marker="s", label="Parking Area")
plt.plot(epochs, no_parking_iou, marker="^", label="No Parking Area")

plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Epoch-wise Class IoU (SegFormer)")
plt.grid(True)
plt.legend()

# 저장
plt.savefig("epochwise_class_iou_fixed.png")
plt.close()
print("✅ 그래프 저장 완료: epochwise_class_iou_fixed.png")
