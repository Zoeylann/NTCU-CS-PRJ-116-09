import time
import os
import requests
import config

class OutputLayer:
    """
    負責所有對外輸出：
      1. 終端機彩色列印
      2. LINE Notify 推播（填入 token 後生效）
      3. 語意化事故摘要（呼叫本地 Ollama SLM）
      4. 防重複通報（同等級警報 30 秒內只發一次）
    """

    # ANSI 終端機顏色
    _COLORS = {
        "HIGH":   "\033[91m",  # 紅
        "MEDIUM": "\033[93m",  # 黃
        "LOW":    "\033[92m",  # 綠
        "RESET":  "\033[0m",
    }

    def __init__(self, location: str):
        self._location      = location
        self._last_alert_time  = 0.0
        self._last_alert_level = ""
        self._cooldown_sec  = 30  # 同等級警報的冷卻秒數

    # ── 主要介面 ───────────────────────────────────
    def alert(self, X, P: float, level: str):
        """
        當風險 P >= RISK_HIGH 時由 main.py 呼叫。
        依序執行：列印 → 生成摘要 → 推播。
        """
        if not self._should_send(level):
            return  # 冷卻中，跳過

        timestamp = time.strftime("%H:%M:%S")
        summary   = self._generate_summary(X, P, level, timestamp)

        self._print_alert(level, P, summary, timestamp)
        self._send_line_notify(level, summary)
        # 若有前端 WebSocket，可在這裡加 self._send_ws(...)

        self._last_alert_time  = time.time()
        self._last_alert_level = level

    def log_status(self, P: float, level: str):
        """
        每次決策都呼叫（LOW / MEDIUM 也會跑到），
        只印一行狀態，不觸發推播。
        """
        color = self._COLORS.get(level, "")
        reset = self._COLORS["RESET"]
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"{color}[{level}]{reset} P={P:.3f}")

    # ── 冷卻判斷 ───────────────────────────────────
    def _should_send(self, level: str) -> bool:
        now     = time.time()
        elapsed = now - self._last_alert_time
        # 等級升高時立刻發（MEDIUM → HIGH 不需等冷卻）
        level_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        if level_rank.get(level, 0) > level_rank.get(self._last_alert_level, 0):
            return True
        return elapsed >= self._cooldown_sec

    # ── 語意摘要（Ollama 本地 SLM）────────────────
    def _generate_summary(self, X, P: float, level: str,
                           timestamp: str) -> str:
        """
        呼叫本地 Ollama（phi3:mini 或 llama3.2:1b）。
        若 Ollama 未安裝或逾時，回傳預設文字。
        """
        weapon_s  = round(float(max(X[0], X[1])), 2)
        pose_s    = round(float(max(X[5], X[6])), 2)
        emotion_s = round(float(max(X[10], X[11])), 2)
        audio_s   = round(float(max(X[15], X[16])), 2)

        prompt = (
            "你是校園安全通報系統。根據以下感測數據，"
            "用繁體中文生成一句簡短通報（30字以內），"
            "不要有前言，直接輸出通報內容：\n"
            f"地點：{self._location}，時間：{timestamp}\n"
            f"武器威脅：{weapon_s}，動作威脅：{pose_s}，"
            f"情緒威脅：{emotion_s}，聲音威脅：{audio_s}\n"
            f"綜合風險：{P:.2f}（{level}）"
        )

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "phi3:mini", "prompt": prompt, "stream": False},
                timeout=5  # 最多等 5 秒，避免拖慢主迴圈
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except requests.exceptions.RequestException:
            pass  # Ollama 未啟動時靜默失敗

        # 預設摘要（Ollama 不可用時）
        dominant = max(
            [("武器", weapon_s), ("動作", pose_s),
             ("情緒", emotion_s), ("聲音", audio_s)],
            key=lambda x: x[1]
        )
        return (f"【{level}】{self._location} {timestamp} "
                f"偵測到異常{dominant[0]}行為，請立即確認。")

    # ── 終端機彩色輸出 ─────────────────────────────
    def _print_alert(self, level: str, P: float,
                     summary: str, timestamp: str):
        color = self._COLORS.get(level, "")
        reset = self._COLORS["RESET"]
        sep   = "=" * 55
        print(f"\n{color}{sep}")
        print(f"  🚨 {level} 警報  P={P:.3f}  {timestamp}")
        print(f"  {summary}")
        print(f"{sep}{reset}\n")

    # ── LINE Notify 推播 ───────────────────────────
    def _send_line_notify(self, level: str, summary: str):
        """
        填入 config.ALERT_WEBHOOK_URL（LINE Notify token）後生效。
        取得 token：https://notify-bot.line.me/my/
        """
        token = getattr(config, "ALERT_WEBHOOK_URL", "")
        if not token:
            return  # 未設定則跳過

        emoji = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️"}.get(level, "")
        message = f"\n{emoji} {summary}"

        try:
            requests.post(
                "https://notify-api.line.me/api/notify",
                headers={"Authorization": f"Bearer {token}"},
                data={"message": message},
                timeout=5
            )
        except requests.exceptions.RequestException as e:
            print(f"[LINE Notify] 發送失敗：{e}")