"""
transcriber.py
Chuyển file audio .mp3 → transcript + word-level timestamps
Dùng Faster-Whisper (chạy hoàn toàn local, không cần API)
"""

from faster_whisper import WhisperModel
import re

# Danh sách filler words phổ biến trong IELTS Speaking (true acoustic fillers)
FILLER_WORDS = {
    "uh", "um", "erm", "er", "ah"
}

_model = None

def _get_model(model_size: str = "base"):
    """Lazy-load model để không tốn RAM khi không dùng."""
    global _model
    if _model is None:
        # device="cpu" đảm bảo chạy trên mọi máy không cần GPU
        # compute_type="int8" giảm RAM, đủ chính xác cho IELTS
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribe file audio, trả về transcript + word timestamps.

    Args:
        audio_path: Đường dẫn file .mp3 / .wav
        model_size: "tiny" | "base" | "small" | "medium" | "large-v3"
                    base = cân bằng tốt giữa tốc độ và độ chính xác
    Returns:
        {
            "transcript": str,
            "words": [{"word": str, "start": float, "end": float}, ...],
            "duration_seconds": float,
            "filler_words": {"uh": 3, "like": 5, ...},
            "filler_count": int,
            "self_corrections": ["i mean", "actually", ...]
        }
    """
    model = _get_model(model_size)

    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
        beam_size=5
    )

    all_words = []
    full_text_parts = []

    for segment in segments:
        full_text_parts.append(segment.text.strip())
        if segment.words:
            for word_obj in segment.words:
                all_words.append({
                    "word": word_obj.word.strip().lower(),
                    "start": round(word_obj.start, 3),
                    "end": round(word_obj.end, 3)
                })

    transcript = " ".join(full_text_parts).strip()
    duration = info.duration if info.duration else (all_words[-1]["end"] if all_words else 0.0)

    # Detect filler words
    filler_counts = _count_filler_words(transcript)

    # Detect self-corrections
    self_corrections = _detect_self_corrections(transcript)

    return {
        "transcript": transcript,
        "words": all_words,
        "duration_seconds": round(duration, 2),
        "filler_words": filler_counts,
        "filler_count": sum(filler_counts.values()),
        "self_corrections": self_corrections,
        "self_correction_count": len(self_corrections)
    }


def _count_filler_words(text: str) -> dict:
    """Đếm số lần xuất hiện từng filler word."""
    text_lower = text.lower()
    counts = {}
    for filler in FILLER_WORDS:
        # Đếm số lần xuất hiện có word boundary
        pattern = r'\b' + re.escape(filler) + r'\b'
        count = len(re.findall(pattern, text_lower))
        if count > 0:
            counts[filler] = count
    return counts


def _detect_self_corrections(text: str) -> list:
    """
    Phát hiện self-correction markers — Band 8+ thường dùng.
    Giám khảo IELTS đánh giá cao khả năng tự sửa lỗi.
    """
    CORRECTION_MARKERS = [
        "i mean", "what i meant", "actually", "sorry",
        "let me rephrase", "or rather", "to be more precise",
        "in other words"
    ]
    text_lower = text.lower()
    found = []
    for marker in CORRECTION_MARKERS:
        if marker in text_lower:
            found.append(marker)
    return found
