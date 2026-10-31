"""
audio_features.py
Trích xuất đặc trưng âm học từ file audio:
  - Fluency features: WPM, articulation_rate, pauses, silent_ratio
  - Pronunciation features: pitch, jitter, shimmer, HNR (via Praat-Parselmouth)
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call


# ─────────────────────────────────────────────────
# FLUENCY FEATURES (từ word timestamps + librosa)
# ─────────────────────────────────────────────────

def extract_fluency_features(words: list, duration_seconds: float,
                              filler_count: int, audio_path: str) -> dict:
    """
    Tính các đặc trưng liên quan đến Fluency & Coherence.

    Args:
        words: List word dicts từ transcriber {"word", "start", "end"}
        duration_seconds: Tổng thời gian audio
        filler_count: Số filler words từ transcriber
        audio_path: Đường dẫn file audio (để detect silence)

    Returns:
        dict các fluency features
    """
    if not words or duration_seconds <= 0:
        return _empty_fluency()

    total_words = len(words)

    # Tính pauses từ khoảng cách giữa các từ
    pauses = _compute_pauses(words)
    short_pauses = [p for p in pauses if 0.5 <= p < 2.0]
    long_pauses  = [p for p in pauses if p >= 2.0]

    # Tính thời gian nói thực (bỏ qua khoảng im)
    speech_duration = _compute_speech_duration(words)
    if speech_duration <= 0:
        speech_duration = duration_seconds

    # Speech Rate (WPM) — tính trên tổng thời gian
    speech_rate_wpm = round((total_words / duration_seconds) * 60, 1)

    # Articulation Rate — tính trên thời gian nói thực (không tính pause)
    articulation_rate = round((total_words / speech_duration) * 60, 1)

    # Silent ratio
    total_pause_time = sum(pauses)
    silent_ratio = round(total_pause_time / duration_seconds, 3)

    # Speaking length (số từ) — Band 9 Part 2 thường ≥ 110 từ
    speaking_length_words = total_words

    return {
        "speech_rate_wpm": speech_rate_wpm,
        "articulation_rate": articulation_rate,
        "pause_count": len(short_pauses) + len(long_pauses),
        "short_pause_count": len(short_pauses),
        "long_pause_count": len(long_pauses),
        "avg_pause_duration": round(np.mean(pauses) if pauses else 0.0, 3),
        "silent_ratio": silent_ratio,
        "speaking_length_words": speaking_length_words,
        "filler_count": filler_count,
    }


def _compute_pauses(words: list) -> list:
    """Tính độ dài các khoảng im giữa các từ liền kề."""
    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= 0.3:  # Chỉ tính pause ≥ 0.3 giây
            pauses.append(round(gap, 3))
    return pauses


def _compute_speech_duration(words: list) -> float:
    """Tính tổng thời gian nói thực (tổng duration của từng từ)."""
    return sum(w["end"] - w["start"] for w in words)


def _empty_fluency() -> dict:
    return {
        "speech_rate_wpm": 0, "articulation_rate": 0,
        "pause_count": 0, "short_pause_count": 0, "long_pause_count": 0,
        "avg_pause_duration": 0, "silent_ratio": 0,
        "speaking_length_words": 0, "filler_count": 0,
    }


# ─────────────────────────────────────────────────
# PRONUNCIATION FEATURES (Praat-Parselmouth)
# ─────────────────────────────────────────────────

def extract_pronunciation_features(audio_path: str) -> dict:
    """
    Trích xuất đặc trưng âm học liên quan đến Pronunciation.
    Dùng Praat-Parselmouth — tiêu chuẩn nghiên cứu ngôn ngữ học.

    Args:
        audio_path: Đường dẫn file audio

    Returns:
        dict các pronunciation features
    """
    try:
        snd = parselmouth.Sound(audio_path)
        return {
            **_extract_pitch_features(snd),
            **_extract_voice_quality(snd),
            **_extract_rhythm(snd),
        }
    except Exception as e:
        print(f"[audio_features] Lỗi Praat: {e}")
        return _empty_pronunciation()


def _extract_pitch_features(snd) -> dict:
    """Pitch (F0) — đo Intonation."""
    try:
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_arr = pitch_values[pitch_values > 0]

        if len(pitch_arr) == 0:
            return {"pitch_mean": 0.0, "pitch_std": 0.0, "pitch_range": 0.0}

        return {
            "pitch_mean": round(float(np.mean(pitch_arr)), 2),
            "pitch_std": round(float(np.std(pitch_arr)), 2),
            "pitch_range": round(float(np.max(pitch_arr) - np.min(pitch_arr)), 2),
        }
    except Exception:
        return {"pitch_mean": 0.0, "pitch_std": 0.0, "pitch_range": 0.0}


def _extract_voice_quality(snd) -> dict:
    """Jitter, Shimmer, HNR — đo chất lượng giọng nói."""
    try:
        # Point process cho jitter/shimmer
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)

        jitter_local = call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer_local = call(
            [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        # HNR (Harmonics-to-Noise Ratio) — cao = giọng rõ
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        return {
            "jitter_local": round(float(jitter_local) if jitter_local else 0.0, 4),
            "shimmer_local": round(float(shimmer_local) if shimmer_local else 0.0, 4),
            "hnr": round(float(hnr) if hnr and not np.isnan(hnr) else 0.0, 2),
        }
    except Exception:
        return {"jitter_local": 0.0, "shimmer_local": 0.0, "hnr": 0.0}


def _extract_rhythm(snd) -> dict:
    """
    Rhythm score — đo tính đều đặn nhịp điệu.
    Dùng entropy của khoảng cách giữa các đỉnh năng lượng.
    Entropy thấp = nhịp điệu đều → thường ở band thấp (đọc thuộc).
    Entropy trung bình = tự nhiên → band cao.
    """
    try:
        y = snd.values[0]
        sr = int(snd.sampling_frequency)

        # RMS energy per frame
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.010 * sr)    # 10ms
        rms = librosa.feature.rms(y=y, frame_length=frame_length,
                                  hop_length=hop_length)[0]

        # Tìm các đỉnh (peaks) trong energy
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(rms, height=np.mean(rms) * 0.5, distance=10)

        if len(peaks) < 3:
            return {"rhythm_score": 0.5}

        # Khoảng cách giữa các đỉnh
        intervals = np.diff(peaks)

        # Entropy — đo tính đa dạng nhịp điệu
        hist, _ = np.histogram(intervals, bins=10, density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log(hist + 1e-9)))

        # Chuẩn hóa về [0, 1]
        rhythm_score = round(min(1.0, entropy / 3.0), 3)

        return {"rhythm_score": rhythm_score}
    except Exception:
        return {"rhythm_score": 0.5}


def _empty_pronunciation() -> dict:
    return {
        "pitch_mean": 0.0, "pitch_std": 0.0, "pitch_range": 0.0,
        "jitter_local": 0.0, "shimmer_local": 0.0, "hnr": 0.0,
        "rhythm_score": 0.5,
    }
