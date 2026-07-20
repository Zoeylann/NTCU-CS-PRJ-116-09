import numpy as np
import sounddevice as sd
import threading
import time
import os
import csv
import config

_yamnet_model   = None
_expert_model   = None
_yamnet_classes = []

_latest_vector  = [0.0] * 5
_vector_lock    = threading.Lock()

# ── 沿用你的類別順序（必須與訓練時的 CATEGORIES 完全一致）──
EXPERT_CATEGORIES = [
    'yell', 'gunshot', 'explosion',
    'shouting', 'screaming', 'Other', 'glassBreaking'
]

# ── 沿用你的日常音安全鎖清單 ──────────────────────────────
IS_NORMAL_LIVE = [
    'Music', 'Musical instrument', 'Singing', 'Guitar',
    'Plucked string instrument', 'Keyboard (musical)', 'Piano',
    'Synthesizer', 'Drum', 'Percussion', 'Speech', 'Conversation',
    'Narration, monologue', 'Babbling', 'Dog', 'Bark',
    'Domestic animals, pets', 'Animal', 'Ding', 'Doorbell', 'Chime'
]


def _load_models():
    global _yamnet_model, _expert_model, _yamnet_classes

    import tensorflow as tf
    import tensorflow_hub as hub

    # Keras 3 相容層（沿用你的修正）
    @tf.keras.utils.register_keras_serializable(package="Custom")
    class CompatibleInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, **kwargs):
            kwargs.pop('optional', None)
            kwargs.pop('ragged', None)
            if 'batch_shape' in kwargs:
                kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
            super().__init__(*args, **kwargs)

    os.environ["TFHUB_CACHE_DIR"] = os.path.join(config.BASE_DIR, "yamnet_cache")
    _yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    try:
        _expert_model = tf.keras.models.load_model(
            config.AUDIO_MODEL_PATH,
            custom_objects={'InputLayer': CompatibleInputLayer}
        )
        print(f"✅ 音訊模型載入完成：{config.AUDIO_MODEL_PATH}")
    except Exception as e:
        raise FileNotFoundError(
            f"❌ 無法載入音訊模型！請確認路徑：{config.AUDIO_MODEL_PATH}\n錯誤：{e}"
        )

    class_map_path = _yamnet_model.class_map_path().numpy()
    with _yamnet_model._tf_api_names_compat[0] if hasattr(_yamnet_model, '_tf_api_names_compat') else open(class_map_path):
        pass
    # 正確讀取 YAMNet 標籤
    import tensorflow as tf
    with tf.io.gfile.GFile(class_map_path) as f:
        for row in csv.DictReader(f):
            _yamnet_classes.append(row['display_name'])

    print("✅ YAMNet 標籤載入完成")


def _preprocess_for_expert(waveform_16k: np.ndarray) -> np.ndarray:
    """沿用你的梅爾頻譜圖預處理，與訓練時保持一致"""
    import librosa
    waveform_22k = librosa.resample(
        waveform_16k, orig_sr=config.AUDIO_SR, target_sr=22050
    )
    mels   = librosa.feature.melspectrogram(y=waveform_22k, sr=22050, n_mels=128)
    db_spec = librosa.power_to_db(mels, ref=np.max)

    if db_spec.shape[1] < 128:
        db_spec = np.pad(db_spec, ((0,0),(0, 128 - db_spec.shape[1])), mode='constant')
    else:
        db_spec = db_spec[:, :128]

    return db_spec[np.newaxis, ..., np.newaxis]


def _process_audio(waveform: np.ndarray) -> list:
    """
    沿用你的雙模型判斷邏輯 + 日常音安全鎖，
    輸出 5 維向量：[第一類, 第二類, 第三類, 音量正規化, 信心度]
    """
    vector = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 分貝計算
    rms        = np.sqrt(np.mean(waveform**2))
    current_db = 20 * np.log10(rms + 1e-9)

    # 沿用你的靜音門檻 -40dB
    if current_db < -40.0:
        vector[2] = 1.0
        vector[3] = max((current_db + 100) / 100, 0.0)
        return vector

    # YAMNet 推論
    scores         = _yamnet_model(waveform)[0]
    yam_mean       = np.mean(scores, axis=0)
    yam_id         = int(np.argmax(yam_mean))
    yam_label      = _yamnet_classes[yam_id]
    yam_conf       = float(yam_mean[yam_id])

    # 專家模型推論
    expert_input   = _preprocess_for_expert(waveform)
    preds          = _expert_model.predict(expert_input, verbose=0)
    idx            = int(np.argmax(preds))
    expert_conf    = float(preds[0][idx])
    expert_label   = EXPERT_CATEGORIES[idx]

    # 日常音安全鎖（沿用你的邏輯）
    is_normal = any(item in yam_label for item in IS_NORMAL_LIVE)

    final_conf = yam_conf

    if is_normal:
        # 安全鎖觸發：強制第三類
        vector[2]  = 1.0
        final_conf = yam_conf

    # 第一類：槍聲、爆炸、尖叫（沿用你的 >0.95 嚴格門檻）
    elif expert_label in ['gunshot', 'explosion', 'screaming'] \
            and expert_conf > 0.95 and not is_normal:
        vector[0]  = expert_conf
        final_conf = expert_conf

    # 第二類：破壞聲、爭執（沿用你的 >0.90 門檻）
    elif expert_label in ['glassBreaking', 'shouting', 'yell'] \
            and expert_conf > 0.90 and not is_normal:
        vector[1]  = expert_conf
        final_conf = expert_conf

    # YAMNet 補漏：尖叫/槍聲/爆炸
    elif any(kw in yam_label for kw in ["Screaming","Crying","Gunshot","Explosion"]) \
            and not is_normal:
        vector[0]  = yam_conf

    # YAMNet 補漏：破碎聲
    elif any(kw in yam_label for kw in ["Breaking","Glass"]) \
            and not is_normal:
        vector[1]  = yam_conf

    else:
        # 背景音
        vector[2]  = max(expert_conf, 0.5)

    # 音量與信心度（第4、5維）
    vector[3] = float(np.clip((current_db + 100) / 100, 0.0, 1.0))
    vector[4] = float(final_conf)

    return vector


def get_vector() -> list:
    """主程式呼叫此函式取得最新音訊向量（非阻塞）"""
    with _vector_lock:
        return _latest_vector.copy()


def start_audio_thread():
    """在背景持續錄音並更新共享向量"""
    _load_models()

    def _loop():
        global _latest_vector
        print("🎙️ 音訊監聽執行緒已啟動")
        while True:
            try:
                audio = sd.rec(
                    int(config.AUDIO_DURATION * config.AUDIO_SR),
                    samplerate=config.AUDIO_SR,
                    channels=1,
                    dtype='float32'
                )
                sd.wait()
                vec = _process_audio(np.squeeze(audio))
                with _vector_lock:
                    _latest_vector = vec
            except Exception as e:
                print(f"[Audio] 錯誤：{e}")
                time.sleep(1)

    threading.Thread(target=_loop, daemon=True).start()