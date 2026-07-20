import os

# ── 場域設定 ──────────────────────────────
DEVICE_ID = "corridor_A"
LOCATION_NAME = "走廊A"
DEVICE_ID_VECTOR = [1, 0, 0]  # 靜態:[1,0,0] 動態:[0,1,0] 危險:[0,0,1]

# ── 模型路徑（相對路徑，方便整組人使用）──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEAPON_MODEL_PATH = os.path.join(BASE_DIR, "models", "weapon_best.pt")
POSE_MODEL_PATH   = os.path.join(BASE_DIR, "models", "yolo11n-pose.pt")
LSTM_MODEL_PATH   = os.path.join(BASE_DIR, "models", "pose_lstm.pth")
FACE_MODEL_PATH   = os.path.join(BASE_DIR, "models", "yolov8n-face.pt")
AUDIO_MODEL_PATH  = os.path.join(BASE_DIR, "models", "my_expert_test.h5")

# ── 攝影機與音訊設定 ────────────────────────
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
AUDIO_SR      = 16000
AUDIO_DURATION = 1.0

# ── 時序同步層與 RingBuffer 參數設定 ─────────
# 根據各模組更新頻率，保留約 2 ~ 5 秒的歷史資料
WEAPON_BUF_SIZE  = 60   # 武器偵測緩衝大小
POSE_BUF_SIZE    = 60   # 姿態分析緩衝大小
EMOTION_BUF_SIZE = 30   # 情緒分析緩衝大小
AUDIO_BUF_SIZE   = 10   # 音訊監聽緩衝大小

# ── 風險閾值 ────────────────────────────────
RISK_HIGH   = 0.8
RISK_MEDIUM = 0.5
VIOLENCE_CONF_THRESHOLD = 0.75  # 姿態模組防誤判門檻

# ── 輸出設定 ────────────────────────────────
LOG_PATH = os.path.join(BASE_DIR, "logs", "events.csv")
ALERT_WEBHOOK_URL = ""  # 填入 LINE Notify token 或留空