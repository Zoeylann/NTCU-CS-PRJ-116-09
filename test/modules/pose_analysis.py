import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ultralytics import YOLO
import config

class PoseLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

CLASS_MAP = {0: "Normal", 1: "Violence", 2: "Help"}

_yolo_model  = None
_lstm_model  = None
_device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_pose_history = {}  # {track_id: deque}

def load_models():
    global _yolo_model, _lstm_model
    if _yolo_model is None:
        _yolo_model = YOLO(config.POSE_MODEL_PATH)
    if _lstm_model is None:
        _lstm_model = PoseLSTM().to(_device)
        _lstm_model.load_state_dict(
            torch.load(config.LSTM_MODEL_PATH, map_location=_device)
        )
        _lstm_model.eval()
        print("✅ 姿態分析模型載入完成")

def get_vector(frame) -> list:
    """
    輸入：BGR 影像幀
    輸出：5 維向量 [暴力分數, 求救分數, 一般分數, 人數(正規化), 最高信心]
    """
    load_models()
    results = _yolo_model.track(frame, persist=True, verbose=False)
    
    vector = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    r = results[0]
    if r.keypoints is None or r.boxes.id is None:
        return vector
    
    keypoints_all = r.keypoints.xyn.cpu().numpy()
    track_ids     = r.boxes.id.int().cpu().tolist()
    
    for i, tid in enumerate(track_ids):
        kpts_flat = keypoints_all[i].flatten()
        if len(kpts_flat) != 34:
            continue
        
        if tid not in _pose_history:
            _pose_history[tid] = deque(maxlen=45)
        _pose_history[tid].append(kpts_flat)
        
        if len(_pose_history[tid]) == 45:
            input_data = torch.FloatTensor(
                np.array([list(_pose_history[tid])])
            ).to(_device)
            
            with torch.no_grad():
                prob      = torch.softmax(_lstm_model(input_data), dim=1)
                conf_val, pred_idx = torch.max(prob, 1)
                label     = CLASS_MAP.get(pred_idx.item(), "Normal")
                confidence = conf_val.item()
                
                # 沿用你們原本的防誤判門檻
                if label == "Violence" and confidence < config.VIOLENCE_CONF_THRESHOLD:
                    label = "Normal"
                
                idx = {"Violence": 0, "Help": 1, "Normal": 2}[label]
                vector[idx] = max(vector[idx], confidence)
    
    # 人數正規化（最多10人）
    vector[3] = min(len(track_ids) / 10.0, 1.0)
    vector[4] = max(vector[:3]) if any(v > 0 for v in vector[:3]) else 0.0
    
    return vector