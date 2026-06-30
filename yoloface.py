import cv2
from ultralytics import YOLO
from deepface import DeepFace
import csv
import time
import os
import requests
from collections import Counter

# --- 自動下載模型邏輯 ---
def ensure_model_exists():
    model_name = "yolov8n-face.pt"
    download_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
    
    if not os.path.exists(model_name):
        print(f"⏳ 找不到 {model_name}，正在自動下載臉部專用模型 (約 6MB)...")
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            with open(model_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✨ 模型下載成功！")
        except Exception as e:
            print(f"❌ 自動下載失敗：{e}")
            print("請檢查網路連線，或手動將模型檔放入資料夾。")
            return False
    return True

# --- 設定區 ---
output_file = r"C:\Users\zoey7\Downloads\test\emotion_data.csv"
save_interval = 1.0  
last_save_time = time.time()

# 確保模型存在後載入
if ensure_model_exists():
    model = YOLO('yolov8n-face.pt')
else:
    print("無法啟動程式，因為缺少模型檔。")
    exit()

# 初始化 CSV
try:
    # 確保資料夾存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Face_ID', 'C1','C2','C3', 'Avg_Intensity', 'Avg_Confidence', 'Dominant_Mode'])
except PermissionError:
    print("❌ 錯誤：請先關閉 CSV 檔案再執行！")

buffers = {} 
cap = cv2.VideoCapture(0)

print("🎯 臉部專用 YOLO 啟動")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    current_time = time.time()

    # 1. 使用臉部專用 YOLO 偵測
    results = model(frame, verbose=False, conf=0.5)
    
    for r in results:
        boxes = r.boxes
        for i, box in enumerate(boxes):
            # 這裡拿到的座標就是精確的臉部區域
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0: continue

            try:
                # 2. DeepFace 分析
                analysis = DeepFace.analyze(face_img, actions=['emotion'], 
                                            enforce_detection=False, 
                                            detector_backend='skip', 
                                            silent=True)
                res = analysis[0]
                face_id = i + 1 
                dom = res['dominant_emotion']
                emo = res['emotion'] # 所有的情緒機率
                conf = float(box.conf[0])
                
                # --- 修正處：使用正確的變數名稱 res ---
                # 取得目前最大類別（dominant_emotion）的原始百分比
                intensity = emo[dom] / 100.0  # 直接從 emo 字典中取得對應標籤的數值

                # 繪製精確的臉部框 (不會再飄到天花板或肩膀)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{face_id} {dom} ({intensity:.2f})", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 存入緩衝區
                if face_id not in buffers:
                    buffers[face_id] = {'emotions': [], 'intensities': [], 'confidences': []}
                
                buffers[face_id]['emotions'].append(dom)
                buffers[face_id]['intensities'].append(intensity)
                buffers[face_id]['confidences'].append(conf)

            except Exception:
                continue

    # --- 每秒結算邏輯 ---
    if current_time - last_save_time >= save_interval:
        timestamp = time.strftime("%H:%M:%S")
        if buffers:
            try:
                with open(output_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    for f_id, data in buffers.items():
                        # 安全防護：確保有資料才計算平均值，防止 ZeroDivisionError
                        if data['emotions'] and data['intensities']:
                            mode_emotion = Counter(data['emotions']).most_common(1)[0][0]
                            avg_intensity = sum(data['intensities']) / len(data['intensities'])
                            avg_conf = sum(data['confidences']) / len(data['confidences'])

                            mapping = {'angry':[1,0,0], 'fear':[1,0,0], 'sad':[0,1,0], 'disgust':[0,1,0], 'happy':[0,0,1], 'surprise':[0,0,1], 'neutral':[0,0,1]}
                            c_list = mapping.get(mode_emotion, [0, 0, 0])
                            writer.writerow([timestamp, f_id, c_list[0], c_list[1], c_list[2], round(avg_intensity, 4), round(avg_conf, 4), mode_emotion])
            except PermissionError:
                pass
            
            buffers = {} 
            last_save_time = current_time

    cv2.imshow('Final Face & Emotion Logger', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()