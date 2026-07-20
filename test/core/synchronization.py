import time
import threading
from collections import deque
import numpy as np
import config  # 💡 串接點：匯入全域專案設定檔

class RingBuffer:
    def __init__(self, max_size=30):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def push(self, item):
        with self.lock:
            self.buffer.append((time.time(), item))
    
    def get_latest(self):
        with self.lock:
            return list(self.buffer)[-1][1] if self.buffer else None
    
    def get_window(self, seconds=1.0):
        """取最近 N 秒內的資料"""
        now = time.time()
        with self.lock:
            return [v for t, v in self.buffer if now - t <= seconds]

class SynchronizationLayer:
    def __init__(self):
        # 💡 串接點：從 config.py 讀取各個模組緩衝區的最大容量 (max_size)
        self.weapon_buf   = RingBuffer(max_size=config.WEAPON_BUF_SIZE)   
        self.pose_buf     = RingBuffer(max_size=config.POSE_BUF_SIZE)
        self.emotion_buf  = RingBuffer(max_size=config.EMOTION_BUF_SIZE)  
        self.audio_buf    = RingBuffer(max_size=config.AUDIO_BUF_SIZE)    
        
        # 💡 串接點：場域的靜態 one-hot 向量也從 config.py 統一管理
        self.device_id_vec = config.DEVICE_ID_VECTOR  # 例如 [1, 0, 0]
    
    def build_feature_vector(self):
        w = self.weapon_buf.get_latest()  or [0.0]*5
        p = self.pose_buf.get_latest()    or [0.0]*5
        e = self.emotion_buf.get_latest() or [0.0]*5
        a = self.audio_buf.get_latest()   or [0.0]*5
        d = self.device_id_vec            # 從 config 讀入的場域向量
        
        X = w + p + e + a + d  # 長度 = 5+5+5+5+3 = 23
        return np.array(X, dtype=np.float32)