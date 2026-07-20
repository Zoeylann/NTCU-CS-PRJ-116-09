import cv2
import numpy as np
import onnxruntime as ort
import os, requests
from collections import Counter
from ultralytics import YOLO
import config

EMOTION_LABELS = ['neutral','happy','surprise','sad','angry','disgust','fear','contempt']
EMOTION_MAPPING = {
    'angry': [1,0,0], 'fear': [1,0,0],       # C1：激動/負面
    'sad':   [0,1,0], 'disgust': [0,1,0],     # C2：抑鬱/潛在風險
    'happy': [0,0,1], 'surprise': [0,0,1],    # C3：正向中性
    'neutral': [0,0,1], 'contempt': [0,0,1]
}

_face_model = None
_ort_sess   = None

def _ensure_models():
    """自動下載模型（保留你們原本的邏輯）"""
    if not os.path.exists(config.FACE_MODEL_PATH):
        r = requests.get("https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt")
        with open(config.FACE_MODEL_PATH, "wb") as f: f.write(r.content)
    
    onnx_path = os.path.join(os.path.dirname(config.BASE_DIR), "models", "emotion_model.onnx")
    if not os.path.exists(onnx_path):
        r = requests.get("https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx")
        with open(onnx_path, "wb") as f: f.write(r.content)
    return onnx_path

def load_models():
    global _face_model, _ort_sess
    if _face_model is None:
        onnx_path   = _ensure_models()
        _face_model = YOLO(config.FACE_MODEL_PATH)
        _ort_sess   = ort.InferenceSession(onnx_path)
        print("✅ 臉部情緒模型載入完成")

def get_vector(frame) -> list:
    """
    輸出：5 維向量 [激動/負面, 抑鬱/風險, 中性/正向, 臉部數量(正規化), 平均信心]
    """
    load_models()
    vector = [0.0, 0.0, 0.0, 0.0, 0.0]
    face_confs = []
    
    results = _face_model(frame, verbose=False, conf=0.5)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            try:
                gray  = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                gray  = cv2.resize(gray, (64, 64)).astype(np.float32)
                blob  = gray.reshape(1, 1, 64, 64)
                
                outputs = _ort_sess.run(None, {'Input3': blob})
                probs   = np.exp(outputs[0][0])
                probs  /= probs.sum()
                
                dom   = EMOTION_LABELS[np.argmax(probs)]
                if dom == 'contempt': dom = 'neutral'
                intensity = float(probs.max())
                conf      = float(box.conf[0])
                
                c_list = EMOTION_MAPPING.get(dom, [0, 0, 1])
                vector[0] = max(vector[0], c_list[0] * intensity)
                vector[1] = max(vector[1], c_list[1] * intensity)
                vector[2] = max(vector[2], c_list[2] * intensity)
                face_confs.append(conf)
                
            except Exception:
                continue
    
    vector[3] = min(len(face_confs) / 5.0, 1.0)
    vector[4] = sum(face_confs) / len(face_confs) if face_confs else 0.0
    return vector