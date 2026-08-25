import time
import os
import threading
import requests
import config

class OutputLayer:
    _COLORS = {
        "HIGH":   "\033[91m",  # 紅
        "MEDIUM": "\033[93m",  # 黃
        "LOW":    "\033[92m",  # 綠
        "RESET":  "\033[0m",
    }

    def __init__(self, location: str):
        self._location = location
        self._last_alert_time = 0.0
        self._last_alert_level = ""
        self._cooldown_sec = 20

    def alert(self, X, P: float, level: str, people_details: list = None):
        """非同步警報發送：不卡頓主迴圈"""
        if not self._should_send(level):
            return

        self._last_alert_time = time.time()
        self._last_alert_level = level
        timestamp = time.strftime("%H:%M:%S")

        threading.Thread(
            target=self._async_alert_worker,
            args=(list(X), P, level, timestamp, people_details or []),
            daemon=True
        ).start()

    def _async_alert_worker(self, X, P: float, level: str, timestamp: str, people_details: list):
        summary = self._generate_summary(X, P, level, timestamp, people_details)
        self._print_alert(level, P, summary, timestamp)
        self._send_line_message(summary)

    def _should_send(self, level: str) -> bool:
        now = time.time()
        elapsed = now - self._last_alert_time
        level_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        if level_rank.get(level, 0) > level_rank.get(self._last_alert_level, 0):
            return True
        return elapsed >= self._cooldown_sec

    def _generate_summary(self, X, P: float, level: str, timestamp: str, people_details: list) -> str:
        """強制要求小型 LLM 輸出純條列結構，無任何開場與結尾廢話"""
        date_str = time.strftime("%Y-%m-%d")
        
        # 整理數值特徵
        threats = []
        if X[0] > 0.4: threats.append(f"持致命武器 (信心度 {X[0]*100:.0f}%)")
        if X[1] > 0.4: threats.append(f"持危險工具 (信心度 {X[1]*100:.0f}%)")
        if X[3] > 0.5 or X[4] > 0.5: threats.append(f"肢體衝突/異常姿態 (信心度 {max(X[3], X[4])*100:.0f}%)")
        if X[9] > 0.5 or X[10] > 0.5: threats.append(f"異常聲響/求救聲 (信心度 {max(X[9], X[10])*100:.0f}%)")
        threat_info = "、".join(threats) if threats else "多模態綜合異常"
        people_info = "；".join(people_details) if people_details else "無特定目標特徵"

        prompt = (
            "你是校安通報中心。請根據以下數據生成純條列通報，禁止任何前言、開場白與結語，格式必須完全符合：\n"
            f"地點：{self._location}\n"
            f"時間：{date_str} {timestamp}\n"
            f"風險等級：{level} (P={P:.2f})\n"
            f"威脅指標：{threat_info}\n"
            f"目標狀態：{people_info}\n\n"
            "請嚴格依據以下範例格式輸出：\n"
            "【🚨 校安緊急警報】\n"
            "📍 地點：[地點]\n"
            "⏰ 時間：[時間]\n"
            "⚠️ 等級：[等級與指數]\n"
            "🎯 威脅：[主要威脅與武器]\n"
            "👥 目標：[人數與動作狀態]\n"
            "👉 指示：[給警衛的一句處置指示]"
        )

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "phi3:mini", "prompt": prompt, "stream": False},
                timeout=2.5
            )
            if resp.status_code == 200:
                result = resp.json().get("response", "").strip()
                if "📍" in result or "•" in result:
                    return result
        except requests.exceptions.RequestException:
            pass

        # 備援規則模板：若 LLM 沒開或逾時，直接輸出極簡條列
        return (
            f"【🚨 校安緊急警報】\n"
            f"📍 地點：CAM-01 ({self._location})\n"
            f"⏰ 時間：{date_str} {timestamp}\n"
            f"⚠️ 等級：{level} (P={P:.2f})\n"
            f"🎯 威脅：{threat_info}\n"
            f"👥 目標：{people_info}\n"
            f"👉 指示：請值班警衛立即攜帶裝備前往現場！"
        )

    def _print_alert(self, level: str, P: float, summary: str, timestamp: str):
        color = self._COLORS.get(level, "")
        reset = self._COLORS["RESET"]
        sep   = "=" * 55
        print(f"\n{color}{sep}")
        print(f"  🚨 {level} 警報觸發  {timestamp}")
        print(f"{summary}")
        print(f"{sep}{reset}\n")

    def _send_line_message(self, message_text: str):
        token = getattr(config, "LINE_CHANNEL_ACCESS_TOKEN", "")
        user_id = getattr(config, "LINE_USER_ID", "")
        if not token or not user_id:
            return

        url = "https://api.line.me/v2/bot/message/push"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        payload = {
            "to": user_id,
            "messages": [{"type": "text", "text": message_text}]
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=4)
            if resp.status_code != 200:
                print(f"[LINE Bot] 發送失敗 ({resp.status_code}): {resp.text}")
        except requests.exceptions.RequestException as e:
            print(f"[LINE Bot] 連線失敗：{e}")