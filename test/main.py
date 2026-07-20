import cv2
import threading
import time
import config

# 匯入各模組
from modules import weapon_detection, pose_analysis, emotion_analysis, audio_detection
from core.synchronization import SynchronizationLayer
from core.decision_fusion  import DecisionFusionLayer
from core.output_layer     import OutputLayer

def main():
    print("🚀 校園安全偵測系統啟動中...")
    
    # 初始化各層
    sync   = SynchronizationLayer(config.DEVICE_VECTOR)
    fusion = DecisionFusionLayer()
    output = OutputLayer(config.LOCATION_NAME)
    logger = DataLogger(config.LOG_PATH)
    
    # 音訊模組先在背景啟動（因為需要 1 秒一個週期）
    audio_detection.start_audio_thread()
    
    # 開啟攝影機
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    print("✅ 系統就緒，按 q 結束")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # ── 各模組推論（影像類每幀都跑）──────────────
        weapon_vec  = weapon_detection.get_vector(frame)
        pose_vec    = pose_analysis.get_vector(frame)
        
        # 臉部偵測較耗時，每 3 幀跑一次
        if frame_count % 3 == 0:
            emotion_vec = emotion_analysis.get_vector(frame)
            sync.update_emotion(emotion_vec)
        
        # 音訊向量從背景執行緒取（非阻塞）
        audio_vec = audio_detection.get_vector()
        
        # ── 推入 Ring Buffer ──────────────────────────
        sync.update_weapon(weapon_vec)
        sync.update_pose(pose_vec)
        sync.update_audio(audio_vec)
        
        # ── 組出 23 維向量並做決策（每 10 幀一次）───────
        if frame_count % 10 == 0:
            X = sync.build_feature_vector()
            P, level, action = fusion.fuse(X)
            
            print(f"[{time.strftime('%H:%M:%S')}] P={P:.3f} | {level}")
            logger.log(X, P, level)
            
            if P >= config.RISK_HIGH:
                output.alert(X, P, level)
        
        # ── 畫面顯示 ────────────────────────────────
        color = (0,0,255) if level=="HIGH" else (0,165,255) if level=="MEDIUM" else (0,255,0)
        cv2.putText(frame, f"Risk: {level}  P={P:.2f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.imshow("Campus Safety Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("系統已關閉")

if __name__ == "__main__":
    main()