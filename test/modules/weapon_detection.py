from ultralytics import YOLO
import config

CLASS_MAPPING = {
    "Pistol": 0, "Knife": 0, "knife_coco": 0,
    "Bat": 1, "baseball bat": 1, "scissors": 1
}
TOTAL_AREA = config.FRAME_WIDTH * config.FRAME_HEIGHT

_model = None  # 延遲載入，避免 import 時就占用 GPU

def load_model():
    global _model
    if _model is None:
        _model = YOLO(config.WEAPON_MODEL_PATH)
        print("✅ 武器偵測模型載入完成")
    return _model

def get_vector(frame) -> list:
    """
    輸入：BGR 影像幀
    輸出：5 維向量 [致命, 危險, 一般, 面積比, 最高信心度]
    無偵測時輸出全零
    """
    model = load_model()
    results = model.predict(source=frame, conf=0.5, verbose=False)
    
    vector = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    for r in results:
        for box in r.boxes:
            label     = model.names[int(box.cls[0])]
            group_idx = CLASS_MAPPING.get(label, 2)
            conf      = float(box.conf[0])
            w, h      = float(box.xywh[0][2]), float(box.xywh[0][3])
            area_ratio = (w * h) / TOTAL_AREA
            
            # one-hot 取最高危險等級
            if group_idx < 2:  # 只記錄危險物品（群組0或1）
                vector[group_idx] = max(vector[group_idx], conf)
            else:
                vector[2] = max(vector[2], conf)
            
            vector[3] = max(vector[3], round(area_ratio, 5))
            vector[4] = max(vector[4], round(conf, 3))
    
    return vector