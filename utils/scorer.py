"""
scorer.py
Rule-based Expert System:
Nhận vào tất cả features → tính 4 sub-scores IELTS (FC, LR, GRA, PR)
Dùng piecewise linear (np.interp) thay hard step → điểm mượt, không nhảy cóc
"""

import numpy as np


def piecewise(value: float, breakpoints: list) -> float:
    """
    Nội suy tuyến tính giữa các điểm ngưỡng.
    breakpoints = [(x0, y0), (x1, y1), ...]
    Nếu value nằm ngoài range → clamp về min/max.
    """
    xs = [p[0] for p in breakpoints]
    ys = [p[1] for p in breakpoints]
    return float(np.clip(np.interp(value, xs, ys), min(ys), max(ys)))


def get_length_penalty(words: int) -> float:
    """
    Phạt điểm nếu độ dài bài nói quá ngắn.
    IELTS Part 2 cần nói liên tục từ 1-2 phút, đạt ít nhất 100-120 từ.
    """
    if words < 50:
        return 1.5
    elif words < 80:
        return 1.0
    elif words < 100:
        return 0.5
    return 0.0


# ─────────────────────────────────────────────────────────────
# 1. FLUENCY & COHERENCE
# ─────────────────────────────────────────────────────────────

def score_fluency_coherence(fluency: dict, coherence: dict) -> float:
    """
    Tính điểm Fluency & Coherence (FC).
    Weights:
      - Articulation Rate:        35% (tốc độ nói thực tế)
      - Speech Rate (WPM):        25% (tốc độ nói tổng thể)
      - Long Pause Count:         15% (khoảng dừng ngắt mạch)
      - Silent Ratio:             15% (tỷ lệ im lặng)
      - Discourse Marker Variety: 10% (độ đa dạng liên từ)
    """
    ar = fluency.get("articulation_rate", 0)
    wpm = fluency.get("speech_rate_wpm", 0)
    lp = fluency.get("long_pause_count", 0)
    sr = fluency.get("silent_ratio", 0)
    dm = coherence.get("discourse_marker_variety", 0)
    words = fluency.get("speaking_length_words", 0)

    # 1. Articulation Rate (WPM thực không tính pause)
    # 120 -> 5.0, 137 -> 5.4, 150 -> 6.0, 166 -> 6.8, 234 -> 9.0
    ar_score = piecewise(ar, [
        (60, 4.0), (100, 4.5), (120, 5.0), (140, 5.5), (160, 6.5), (180, 7.5), (200, 8.5), (220, 9.0)
    ]) * 0.35

    # 2. Speech Rate (WPM tổng)
    # 96.6 -> 4.66, 108.6 -> 4.96, 140 -> 6.5, 165 -> 7.75, 189 -> 9.0
    wpm_score = piecewise(wpm, [
        (70, 4.0), (90, 4.5), (110, 5.0), (130, 6.0), (150, 7.0), (170, 8.0), (190, 9.0)
    ]) * 0.25

    # 3. Long Pause Count
    lp_score = piecewise(lp, [
        (0, 9.0), (1, 7.5), (2, 6.0), (3, 5.0), (4, 4.0), (6, 3.0)
    ]) * 0.15

    # 4. Silent Ratio
    sr_score = piecewise(sr, [
        (0.05, 9.0), (0.10, 8.0), (0.15, 6.5), (0.20, 5.5), (0.30, 4.5), (0.45, 3.5)
    ]) * 0.15

    # 5. Discourse Marker Variety
    dm_score = piecewise(dm, [
        (0.0, 4.0), (0.17, 5.0), (0.33, 6.0), (0.50, 7.5), (0.67, 8.5), (0.83, 9.0)
    ]) * 0.10

    raw = ar_score + wpm_score + lp_score + sr_score + dm_score
    
    # Áp dụng hình phạt độ dài
    penalty = get_length_penalty(words)
    
    # Áp dụng hình phạt filler count (True hesitation chỉ gồm uh, um, er, ah)
    fc = fluency.get("filler_count", 0)
    filler_penalty = 0.0
    if words >= 100:
        if fc > 12:
            filler_penalty = 0.5
    else:
        if fc > 6:
            filler_penalty = 0.5

    return round(float(np.clip(raw - penalty - filler_penalty, 4.0, 9.0)), 2)


# ─────────────────────────────────────────────────────────────
# 2. LEXICAL RESOURCE
# ─────────────────────────────────────────────────────────────

def score_lexical_resource(lexical: dict) -> float:
    """
    Tính điểm Lexical Resource (LR).
    Weights:
      - Rare Word Ratio:          50% (Độ rộng vốn từ ít gặp)
      - MATTR (length adjusted):  30% (Độ đa dạng từ vựng)
      - Advanced Vocab (AWL):     20% (Từ vựng học thuật)
    """
    rare = lexical.get("rare_word_ratio", 0)
    mattr = lexical.get("mattr", 0)
    awl = lexical.get("advanced_vocab_ratio", 0)
    words = lexical.get("total_words", 0)

    # 1. Rare Word Ratio
    # 0.021 -> 5.73, 0.007 -> 4.7, 0.039 -> 7.1, 0.030 -> 6.5, 0.031 -> 6.56
    rare_score = piecewise(rare, [
        (0.00, 4.0), (0.015, 5.0), (0.03, 6.5), (0.045, 8.0), (0.06, 9.0)
    ]) * 0.50

    # 2. MATTR điều chỉnh theo độ dài
    adj_mattr = mattr
    if words < 80:
        adj_mattr = mattr * (words / 80.0)
    
    mattr_score = piecewise(adj_mattr, [
        (0.20, 4.0), (0.35, 5.0), (0.45, 6.0), (0.55, 7.0), (0.65, 8.0), (0.70, 9.0)
    ]) * 0.30

    # 3. AWL Ratio
    awl_score = piecewise(awl, [
        (0.00, 4.5), (0.005, 5.5), (0.01, 7.0), (0.015, 8.0), (0.02, 9.0)
    ]) * 0.20

    # Phạt lặp từ quá nhiều
    repetition_penalty = 0.0
    max_rep = lexical.get("max_word_repetition", 0)
    if words > 0:
        if max_rep / words > 0.10:
            repetition_penalty = 0.5
        elif max_rep / words > 0.06:
            repetition_penalty = 0.25

    raw = rare_score + mattr_score + awl_score - repetition_penalty
    penalty = get_length_penalty(words)

    return round(float(np.clip(raw - penalty, 4.0, 9.0)), 2)


# ─────────────────────────────────────────────────────────────
# 3. GRAMMATICAL RANGE & ACCURACY
# ─────────────────────────────────────────────────────────────

def score_grammatical_range_accuracy(grammar: dict) -> float:
    """
    Tính điểm Grammatical Range & Accuracy (GRA).
    Weights:
      - Grammar Error Rate:       50% (Độ chính xác ngữ pháp)
      - Complex Clause Ratio:     50% (Độ phức tạp câu)
    """
    er = grammar.get("grammar_error_rate", 0)
    cc = grammar.get("complex_clause_ratio", 0)
    words = grammar.get("sentence_count", 0) * 15 # Uớc lượng số từ thô nếu không truyền

    # 1. Grammar Error Rate (lỗi / 100 từ)
    er_score = piecewise(er, [
        (0.0, 9.0), (0.5, 8.5), (1.0, 7.5), (1.5, 6.5), (2.0, 6.0), (3.0, 5.0), (4.5, 4.0)
    ]) * 0.50

    # 2. Complex Clause Ratio (cục bộ trên câu)
    # Giới hạn cc trong khoảng hợp lý. Do spaCy đếm nhãn dep phụ, cc tỷ lệ cao hơn số câu thực tế.
    cc_clamped = min(1.0, cc / 2.0)
    cc_score = piecewise(cc_clamped, [
        (0.0, 4.5), (0.2, 5.5), (0.4, 6.5), (0.6, 7.5), (0.8, 8.5), (1.0, 9.0)
    ]) * 0.50

    raw = er_score + cc_score
    
    # Lấy số từ thực tế nếu có trong dict
    total_words = grammar.get("total_words", words)
    penalty = get_length_penalty(total_words)

    return round(float(np.clip(raw - penalty, 4.0, 9.0)), 2)


# ─────────────────────────────────────────────────────────────
# 4. PRONUNCIATION
# ─────────────────────────────────────────────────────────────

def score_pronunciation(audio_pron: dict, fluency: dict) -> float:
    """
    Tính điểm Pronunciation (PR).
    Lưu ý: Acoustic features (jitter, shimmer, HNR) bị phụ thuộc nhiều vào chất lượng mic.
    Vì thế ta kết hợp gián tiếp với Fluency (người phát âm tốt thường nói trôi chảy hơn).
    Weights:
      - Articulation Rate:        40% (Độ rõ và mượt của phát âm âm tiết)
      - Pitch Std (Intonation):   30% (Độ đa dạng ngữ điệu)
      - Fluency (FC raw score):   30% (Sự liên kết âm)
    """
    ar = fluency.get("articulation_rate", 0)
    pitch_std = audio_pron.get("pitch_std", 0)
    words = fluency.get("speaking_length_words", 0)

    # 1. Articulation Rate
    ar_score = piecewise(ar, [
        (80, 4.0), (100, 4.5), (120, 5.0), (140, 6.0), (160, 7.0), (180, 8.0), (200, 9.0)
    ]) * 0.40

    # 2. Pitch Std (Intonation)
    # Giới hạn pitch std cực cao (có thể là lỗi tracking hoặc gượng gạo)
    if pitch_std > 80:
        pitch_score = 7.5
    else:
        pitch_score = piecewise(pitch_std, [
            (25, 4.0), (35, 5.5), (45, 7.0), (55, 8.5), (70, 9.0)
        ])
    pitch_score_weighted = pitch_score * 0.30

    # 3. Fluency base (tính FC thô không phạt độ dài để tránh double penalty)
    # Ta tính nhanh một raw FC để làm nền tảng
    lp = fluency.get("long_pause_count", 0)
    sr = fluency.get("silent_ratio", 0)
    lp_score = piecewise(lp, [(0, 9.0), (2, 6.0), (4, 4.0)])
    sr_score = piecewise(sr, [(0.05, 9.0), (0.20, 6.0), (0.35, 4.0)])
    raw_fc = (lp_score + sr_score) / 2.0
    fc_base = raw_fc * 0.30

    raw = ar_score + pitch_score_weighted + fc_base
    penalty = get_length_penalty(words)

    return round(float(np.clip(raw - penalty, 4.0, 9.0)), 2)
