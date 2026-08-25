import numpy as np
import json
import csv
import os
import time
import config

class DecisionFusionLayer:
    """
    接收 15 維特徵向量，輸出風險機率 P 與等級。

    索引對照：
      X[0:3]   = 武器  [致命, 危險工具, 信心]
      X[3:6]   = 姿態  [暴力, 求救, 信心]
      X[6:9]   = 情緒  [激動, 抑鬱, 信心]
      X[9:12]  = 音訊  [緊急, 破壞, 信心]
      X[12:15] = 場域  [靜態, 動態, 危險]
    """

    FEATURE_LABELS = [
        "weapon_lethal", "weapon_danger",
        "pose_violence", "pose_help",
        "emotion_agitated", "emotion_depressed",
        "audio_emergency", "audio_damage",
    ]

    _DEFAULT_WEIGHTS = np.array([
        0.35, 0.20,
        0.25, 0.12,
        0.10, 0.06,
        0.15, 0.08,
    ])

    _SCENE_MULTIPLIER = {
        "static":  1.00,
        "dynamic": 0.90,
        "danger":  1.15,
    }

    MOTION_FAST_THRESHOLD = 0.5

    def __init__(self):
        self._weights = self._load_weights()
        self._scene = "static"

    def _load_weights(self) -> np.ndarray:
        if os.path.exists(config.WEIGHTS_PATH):
            try:
                with open(config.WEIGHTS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                weights = np.array([data[label] for label in self.FEATURE_LABELS])
                print(f"✅ 已載入訓練權重：{config.WEIGHTS_PATH}")
                return weights
            except Exception as e:
                print(f"⚠️ 讀取訓練權重失敗，改用預設值：{e}")
        print("ℹ️ 尚未找到訓練權重檔，使用內建手動預設值")
        return self._DEFAULT_WEIGHTS.copy()

    def reload_weights(self):
        self._weights = self._load_weights()

    def _select_scene(self, device_vec: list) -> str:
        if device_vec == [0, 0, 1]:
            return "danger"
        elif device_vec == [0, 1, 0]:
            return "dynamic"
        else:
            return "static"

    def _fine_grained_scores(self, X: np.ndarray) -> np.ndarray:
        return np.array([
            X[0]  * X[2],
            X[1]  * X[2],
            X[3]  * X[5],
            X[4]  * X[5],
            X[6]  * X[8],
            X[7]  * X[8],
            X[9]  * X[11],
            X[10] * X[11],
        ])

    def _check_combination_rules(self, weapon_detected, weapon_lethal, pose_violent, is_fast) -> float:
        """
        調整說明：規則4(純暴力動作、無武器)的保底值從0.45提高到0.55，
        讓姿態模組一旦判定暴力，系統整體反應更靈敏、更容易達到中風險以上。
        """
        RULES = [
            (weapon_lethal,                                0.75),
            (weapon_detected and pose_violent and is_fast,  0.85),
            (weapon_detected and pose_violent,               0.75),
            (weapon_detected and not pose_violent,           0.50),
            (pose_violent and not weapon_detected,           0.45),
        ]

        best_floor = None
        for condition, floor_value in RULES:
            if condition:
                if best_floor is None or floor_value > best_floor:
                    best_floor = floor_value
        return best_floor

    def fuse(self, X: np.ndarray, motion_score: float = None) -> tuple[float, str, str]:
        device_vec = list(X[12:15].astype(int))
        self._scene = self._select_scene(device_vec)

        features = self._fine_grained_scores(X)
        raw = float(np.dot(self._weights, features))

        module_max = np.array([
            max(features[0], features[1]),
            max(features[2], features[3]),
            max(features[4], features[5]),
            max(features[6], features[7]),
        ])
        triggered = int(np.sum(module_max > 0.4))
        if triggered >= 3:
            raw *= 1.35
        elif triggered == 2:
            raw *= 1.15

        weapon_detected = max(X[0], X[1]) > 0.4
        weapon_lethal    = X[0] > 0.5
        pose_violent     = X[3] > 0.5
        is_fast = (motion_score is not None) and (motion_score > self.MOTION_FAST_THRESHOLD)

        rule_floor = self._check_combination_rules(weapon_detected, weapon_lethal, pose_violent, is_fast)
        if rule_floor is not None:
            raw = max(raw, rule_floor)

        raw *= self._SCENE_MULTIPLIER[self._scene]

        P = float(np.clip(raw, 0.0, 1.0))

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

    def fuse_person(self, person, scene: str = "static") -> tuple[float, str]:
        weapon_lethal_score = person.weapon_confidence if person.weapon_group_idx == 0 else 0.0
        weapon_danger_score = person.weapon_confidence if person.weapon_group_idx == 1 else 0.0
        pose_violence_score = person.pose_confidence if person.pose_label == "Violence" else 0.0
        pose_help_score     = person.pose_confidence if person.pose_label == "Help" else 0.0
        emotion_agit_score  = person.emotion_confidence if person.emotion_class_idx == 0 else 0.0
        emotion_dep_score   = person.emotion_confidence if person.emotion_class_idx == 1 else 0.0

        features = np.array([
            weapon_lethal_score, weapon_danger_score,
            pose_violence_score, pose_help_score,
            emotion_agit_score, emotion_dep_score,
        ])

        raw = float(np.dot(self._weights[:6], features))

        weapon_detected = max(weapon_lethal_score, weapon_danger_score) > 0.4
        weapon_lethal    = weapon_lethal_score > 0.5
        pose_violent     = pose_violence_score > 0.5
        is_fast = getattr(person, "motion_score", 0.0) > self.MOTION_FAST_THRESHOLD

        rule_floor = self._check_combination_rules(weapon_detected, weapon_lethal, pose_violent, is_fast)
        if rule_floor is not None:
            raw = max(raw, rule_floor)

        raw *= self._SCENE_MULTIPLIER.get(scene, 1.0)

        P = float(np.clip(raw, 0.0, 1.0))

        if P > config.RISK_HIGH:
            level = "HIGH"
        elif P > config.RISK_MEDIUM:
            level = "MEDIUM"
        else:
            level = "LOW"

        return P, level

    def explain(self, X: np.ndarray) -> dict:
        features = self._fine_grained_scores(X)
        return {
            "scores":   {k: round(float(v), 3) for k, v in zip(self.FEATURE_LABELS, features)},
            "weights":  {k: round(float(v), 3) for k, v in zip(self.FEATURE_LABELS, self._weights)},
            "scene":    self._scene,
        }

    def log_decision(self, X: np.ndarray, P: float, level: str):
        features = self._fine_grained_scores(X)
        os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
        file_exists = os.path.exists(config.LOG_PATH)

        with open(config.LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp'] + self.FEATURE_LABELS + ['P', 'level', 'scene', 'human_label'])
            writer.writerow(
                [time.strftime('%Y-%m-%d %H:%M:%S')]
                + [round(float(v), 4) for v in features]
                + [round(P, 3), level, self._scene, '']
            )