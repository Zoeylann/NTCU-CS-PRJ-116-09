import cv2
import numpy as np
from ultralytics import YOLO
import time  # 引入時間模組來計算 FPS

def run_live():
    model_path = r"C:\Users\timmy\OneDrive\桌面\Ultimate_Weapon_Dataset\runs\detect\train\weights\best.pt"
    model = YOLO(model_path)

    class_mapping = {
        "Pistol": 0, "Knife": 0, "Sword": 0, "Rifle": 0,
        "Bat": 1, "Hammer": 1, "Scissors": 1
    }

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    total_area = 1280 * 720
    
    # 初始化時間變數
    prev_time = 0

    print("🚀 啟動即時辨識... 按 'q' 結束")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # 計算 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        results = model.predict(source=frame, conf=0.5, verbose=False)

        for r in results:
            annotated_frame = r.plot() 

            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                group_idx = class_mapping.get(label, 2)
                
                one_hot = [0, 0, 0]
                one_hot[group_idx] = 1
                
                w, h = box.xywh[0][2].item(), box.xywh[0][3].item()
                area_ratio = (w * h) / total_area
                conf = box.conf[0].item()

                output_vector = one_hot + [round(float(area_ratio), 5), round(float(conf), 2)]
                print(f"[{label}] -> {output_vector}")

            # 在畫面上顯示 FPS 數字 (綠色字體)
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Weapon Detection Live", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live()