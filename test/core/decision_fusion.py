import numpy as np
import config

class DecisionFusionLayer:
    """
    接收 23 維特徵向量，輸出風險機率 P 與等級。
    
    索引對照：
      X[0:5]  = 物件  [致命, 危險, 一般, 面積比, 信心]
      X[5:10] = 姿態  [暴力, 求救, 一般, 人數, 信心]
      X[10:15]= 情緒  [激動, 抑鬱, 中性, 人臉數, 信心]
      X[15:20]= 音訊  [緊急, 破壞, 背景, 音量, 信心]
      X[20:23]= 場域  [靜態, 動態, 危險]
    """

    # 基礎權重（四個模組）
    _BASE_WEIGHTS = {
        "static":  np.array([0.40, 0.25, 0.20, 0.15]),  # 教室/走廊
        "dynamic": np.array([0.30, 0.40, 0.10, 0.20]),  # 操場/健身房
        "danger":  np.array([0.35, 0.35, 0.10, 0.20]),  # 頂樓/機房
    }

    def __init__(self):
        self._weights = self._BASE_WEIGHTS["static"].copy()

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1 / (1 + np.exp(-x)))

    def _select_weights(self, device_vec: list):
        """根據場域編碼動態切換權重"""
        if device_vec == [0, 0, 1]:
            self._weights = self._BASE_WEIGHTS["danger"].copy()
        elif device_vec == [0, 1, 0]:
            self._weights = self._BASE_WEIGHTS["dynamic"].copy()
        else:
            self._weights = self._BASE_WEIGHTS["static"].copy()

    def _module_scores(self, X: np.ndarray) -> np.ndarray:
        """
        從 23 維向量中萃取四個模組各自的「威脅分數」（0~1）。
        取每個模組的最高危險維度，而非單純平均。
        """
        weapon_score  = float(max(X[0], X[1]))       # 致命 or 危險利器
        pose_score    = float(max(X[5], X[6]))        # 暴力 or 求救
        emotion_score = float(max(X[10], X[11]))      # 激動 or 抑鬱
        audio_score   = float(max(X[15], X[16]))      # 緊急 or 破壞聲
        return np.array([weapon_score, pose_score, emotion_score, audio_score])

    def fuse(self, X: np.ndarray) -> tuple[float, str, str]:
        """
        主要介面：輸入 23 維向量，回傳 (P, level, action)
        
        P      : 0.0 ~ 1.0 的連續風險機率
        level  : "HIGH" / "MEDIUM" / "LOW"
        action : 對應的系統動作描述
        """
        device_vec = list(X[20:23].astype(int))
        self._select_weights(device_vec)

        scores = self._module_scores(X)

        # 加權融合
        raw = float(np.dot(self._weights, scores))

        # ── 多模態同時觸發加成 ──────────────────────
        # 三個以上模組同時異常 → 提升分數（降低誤報）
        triggered = int(np.sum(scores > 0.4))
        if triggered >= 3:
            raw *= 1.35
        elif triggered == 2:
            raw *= 1.15

        # ── 武器直接觸發高風險 ─────────────────────
        # 偵測到致命武器（X[0] > 0.5）無論其他模組直接拉到高風險區
        if X[0] > 0.5:
            raw = max(raw, 0.75)

        # ── Sigmoid 映射到 0~1 ─────────────────────
        # 中心偏移 0.45，斜率 8 → 讓 0.45 附近的變化最敏感
        P = self._sigmoid((raw - 0.45) * 8)
        P = float(np.clip(P, 0.0, 1.0))

        # ── 風險等級判定 ───────────────────────────
        if P > config.RISK_HIGH:
            level  = "HIGH"
            action = "即時發送警報、鎖定畫面、通報校安中心"
        elif P > config.RISK_MEDIUM:
            level  = "MEDIUM"
            action = "標記關注對象、通知警衛注意"
        else:
            level  = "LOW"
            action = "持續監控"

        return P, level, action

    def explain(self, X: np.ndarray) -> dict:
        """
        回傳各模組分數的詳細分解（方便除錯 / 前端顯示）
        """
        scores = self._module_scores(X)
        labels = ["weapon", "pose", "emotion", "audio"]
        return {
            "scores":   {k: round(float(v), 3) for k, v in zip(labels, scores)},
            "weights":  {k: round(float(v), 3) for k, v in zip(labels, self._weights)},
            "triggered_modules": int(np.sum(scores > 0.4)),
        }