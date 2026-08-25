import time
import threading
from collections import deque
import numpy as np
import config

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

class SynchronizationLayer:
    def __init__(self):
        self.weapon_buf   = RingBuffer(max_size=config.WEAPON_BUF_SIZE)
        self.pose_buf     = RingBuffer(max_size=config.POSE_BUF_SIZE)
        self.emotion_buf  = RingBuffer(max_size=config.EMOTION_BUF_SIZE)
        self.audio_buf    = RingBuffer(max_size=config.AUDIO_BUF_SIZE)

        # 存放最新的實體空間結構
        self.latest_persons = []
        self.latest_weapons = []
        self.latest_faces   = []

    def build_feature_vector(self):
        """
        組裝 15 維特徵向量（相容舊版 main.py）：
          X[0:3]   = 武器  [致命, 危險工具, 信心]
          X[3:6]   = 姿態  [暴力, 求救, 信心]
          X[6:9]   = 情緒  [激動, 抑鬱, 信心]
          X[9:12]  = 音訊  [緊急, 破壞, 信心]
          X[12:15] = 場域  [靜態, 動態, 危險]
        """
        w = self.weapon_buf.get_latest()  or [0.0]*3
        p = self.pose_buf.get_latest()    or [0.0]*3
        e = self.emotion_buf.get_latest() or [0.0]*3
        a = self.audio_buf.get_latest()   or [0.0]*3
        d = config.DEVICE_ID_VECTOR

        X = w + p + e + a + d
        return np.array(X, dtype=np.float32)
    
    def update_spatial_data(self, persons=None, weapons=None, faces=None):
        if persons is not None:
            self.latest_persons = persons
        if weapons is not None:
            self.latest_weapons = weapons
        if faces is not None:
            self.latest_faces = faces

    def update_audio(self, vector):
        if vector is not None:
            self.audio_buf.push(vector)

    # ── 向下相容舊版介面 ─────────────────────────────
    def update_weapon(self, vector):
        if vector is not None:
            self.weapon_buf.push(vector)

    def update_pose(self, vector):
        if vector is not None:
            self.pose_buf.push(vector)

    def update_emotion(self, vector):
        if vector is not None:
            self.emotion_buf.push(vector)

    def build_instance_graph(self):
        """
        以人為節點構建圖結構：
        回傳: (node_features, adj_matrix, track_ids, person_centers)
          - node_features: [N, 8]
            [致命武器, 危險工具, 暴力姿態, 求救姿態, 激動情緒, 抑鬱情緒, 音訊緊急, 音訊破壞]
          - adj_matrix: [N, N] 依據空間距離衰減
        """
        audio_vec = self.audio_buf.get_latest() or [0.0] * 5
        audio_emer = audio_vec[0] * audio_vec[4]
        audio_dam  = audio_vec[1] * audio_vec[4]

        N = len(self.latest_persons)
        if N == 0:
            return None, None, [], []

        node_features = []
        person_centers = []
        track_ids = []

        for p in self.latest_persons:
            tid = p['track_id']
            box = p['box']
            kpts = p['keypoints'] # [17, 2]
            track_ids.append(tid)

            center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
            person_centers.append(center)

            # ── 1. 武器綁定（手腕關鍵點 9: 左手腕, 10: 右手腕） ──
            lw = kpts[9]  if len(kpts) > 9  else np.array([0, 0])
            rw = kpts[10] if len(kpts) > 10 else np.array([0, 0])

            weapon_lethal = 0.0
            weapon_danger = 0.0

            for w in self.latest_weapons:
                wbox = w['box']
                w_center = np.array([(wbox[0] + wbox[2]) / 2.0, (wbox[1] + wbox[3]) / 2.0])
                
                # 計算手腕與武器距離
                d_lw = np.linalg.norm(lw - w_center) if lw[0] > 0 else 999.0
                d_rw = np.linalg.norm(rw - w_center) if rw[0] > 0 else 999.0
                min_wrist_dist = min(d_lw, d_rw)

                # 武器中心落在身體框內也算持械
                in_body = (box[0] <= w_center[0] <= box[2]) and (box[1] <= w_center[1] <= box[3])

                if min_wrist_dist < 80 or in_body:
                    if w['group_idx'] == 0:
                        weapon_lethal = max(weapon_lethal, w['conf'])
                    elif w['group_idx'] == 1:
                        weapon_danger = max(weapon_danger, w['conf'])

            # ── 2. 情緒綁定（臉部中心落在頭部區間） ──
            h = box[3] - box[1]
            emo_agitated  = 0.0
            emo_depressed = 0.0

            for f in self.latest_faces:
                fbox = f['box']
                fc_x = (fbox[0] + fbox[2]) / 2.0
                fc_y = (fbox[1] + fbox[3]) / 2.0

                if (box[0] <= fc_x <= box[2]) and (box[1] <= fc_y <= box[1] + 0.45 * h):
                    e_vec = f['emotion_vec']
                    emo_agitated  = max(emo_agitated, e_vec[0] * f['conf'])
                    emo_depressed = max(emo_depressed, e_vec[1] * f['conf'])

            # ── 3. 姿態分數 ──
            pose_violence = p['pose_scores'][0]
            pose_help     = p['pose_scores'][1]

            # 節點 8 維特徵
            node_feat = [
                weapon_lethal, weapon_danger,
                pose_violence, pose_help,
                emo_agitated, emo_depressed,
                audio_emer, audio_dam
            ]
            node_features.append(node_feat)

        node_features = np.array(node_features, dtype=np.float32) # [N, 8]

        # ── 4. 構建鄰接矩陣（高斯距離權重） ──
        adj_matrix = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i == j:
                    adj_matrix[i][j] = 1.0 # 自連接 (Self-loop)
                else:
                    dist = np.linalg.norm(person_centers[i] - person_centers[j])
                    # 180 像素以內具備高度關聯性
                    adj_matrix[i][j] = np.exp(-(dist ** 2) / (2 * (180.0 ** 2)))

        return node_features, adj_matrix, track_ids, person_centers