import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ultralytics import YOLO

# 1. 定義 LSTM 模型架構 (必須放在最前面以避免 NameError)
class PoseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Size)
        _, (h_n, _) = self.lstm(x)
        # 取得最後一個時間點的輸出
        return self.fc(h_n[-1])

# 2. 初始化設備與載入模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"目前使用設備: {device}")

# 載入 YOLOv11 姿勢估計模型
model_yolo = YOLO('yolo11n-pose.pt')

# 載入自定義 LSTM 模型 (輸入 34 個座標, 隱藏層 64, 類別 3)
lstm_model = PoseLSTM(34, 64, 3).to(device)
lstm_model.load_state_dict(torch.load('pose_lstm.pth'))
lstm_model.eval()

# 3. 設定對照表
class_map = {0: "Normal", 1: "Violence", 2: "Help"}
color_map = {
    "Normal": (0, 255, 0),      # 綠色
    "Violence": (0, 0, 255),    # 紅色
    "Help": (0, 255, 255)       # 黃色
}

# 4. 多目標記憶字典：為每個 Track ID 建立獨立的 deque
multi_pose_history = {} 

# 開啟攝影機
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # 執行 YOLO 追蹤 (persist=True 是 ID 追蹤的關鍵)
    results = model_yolo.track(frame, persist=True, verbose=False)
    
    # 檢查是否有偵測到人且具備 ID
    if results[0].keypoints is not None and results[0].boxes.id is not None:
        # 取得所有人的關鍵點 (格式: 人數 x 17點 x 2座標)
        keypoints_all = results[0].keypoints.xyn.cpu().numpy() 
        # 取得所有人的 Track ID
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # 取得所有人的邊框位置 (用於繪圖)
        boxes_all = results[0].boxes.xyxy.cpu().numpy()
        
        # 遍歷畫面上出現的每一個人
        for i, tid in enumerate(track_ids):
            # 取得該 ID 的 34 個座標點 (17*2)
            kpts_flat = keypoints_all[i].flatten()
            
            if len(kpts_flat) == 34:
                # 如果是新出現的 ID，建立專屬緩衝區
                if tid not in multi_pose_history:
                    multi_pose_history[tid] = deque(maxlen=15)
                
                multi_pose_history[tid].append(kpts_flat)
                
                # 當該 ID 積滿 15 幀，進行動作推論
                if len(multi_pose_history[tid]) == 15:
                    # 轉換為三維張量 (1, 15, 34)
                    input_data = torch.FloatTensor(np.array([list(multi_pose_history[tid])])).to(device)
                    
                    with torch.no_grad():
                        output = lstm_model(input_data)
                        prob = torch.softmax(output, dim=1)
                        conf_val, pred_idx = torch.max(prob, 1)
                        
                        # 取得對應標籤與顏色
                        label_name = class_map.get(pred_idx.item(), "Unknown")
                        display_color = color_map.get(label_name, (255, 255, 255))
                        
                        # 繪製資訊到畫面上 (跟隨該人物的框位置)
                        x1, y1, x2, y2 = boxes_all[i]
                        display_text = f"ID:{tid} {label_name} {conf_val.item()*100:.1f}%"
                        
                        # 繪製文字背景陰影 (增加辨識度)
                        cv2.putText(frame, display_text, (int(x1), int(y1) - 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                        # 繪製彩色文字
                        cv2.putText(frame, display_text, (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)

    # 顯示視窗
    cv2.imshow('Multi-Person AI Monitor (LSTM)', frame)
    
    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()