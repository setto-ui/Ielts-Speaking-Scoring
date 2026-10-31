"""
feedback.py
Tạo feedback động, cụ thể và có số liệu thực — không chung chung.
Mỗi tiêu chí có feedback riêng kèm ví dụ cụ thể từ bài nói.
"""

from typing import Optional


def generate_feedback(fluency: dict, lexical: dict,
                      grammar: dict, coherence: dict,
                      sub_scores: dict) -> dict:
    """
    Tạo feedback cụ thể cho từng tiêu chí IELTS.

    Returns:
        {
            "fluency_coherence": str,
            "lexical_resource": str,
            "grammatical_range_accuracy": str,
            "pronunciation": str,
            "strengths": [str, ...],
            "improvements": [str, ...]
        }
    """
    fc_score  = sub_scores.get("fluency_coherence", 0)
    lr_score  = sub_scores.get("lexical_resource", 0)
    gra_score = sub_scores.get("grammatical_range_accuracy", 0)
    pr_score  = sub_scores.get("pronunciation", 0)

    feedback = {
        "fluency_coherence":           _fb_fluency(fluency, coherence, fc_score),
        "lexical_resource":            _fb_lexical(lexical, lr_score),
        "grammatical_range_accuracy":  _fb_grammar(grammar, gra_score),
        "pronunciation":               _fb_pronunciation(sub_scores, pr_score),
        "strengths":                   _get_strengths(fluency, lexical, grammar, coherence, sub_scores),
        "improvements":                _get_improvements(fluency, lexical, grammar, coherence, sub_scores),
    }
    return feedback


# ──────────────────────────────────────────────────────────
# FLUENCY & COHERENCE FEEDBACK
# ──────────────────────────────────────────────────────────

def _fb_fluency(fluency: dict, coherence: dict, score: float) -> str:
    parts = []

    # Filler words
    fw = fluency.get("filler_words", {})
    fc = fluency.get("filler_count", 0)
    if fc > 0:
        detail = ", ".join(f"'{k}' ({v} lần)" for k, v in
                           sorted(fw.items(), key=lambda x: -x[1]))
        parts.append(
            f"Bạn sử dụng {fc} filler word(s): {detail}. "
            f"Thay bằng khoảng dừng có chủ đích (deliberate pause) sẽ tự nhiên hơn."
        )
    else:
        parts.append("Rất ít filler words — đây là điểm mạnh về fluency.")

    # Long pauses
    lp = fluency.get("long_pause_count", 0)
    if lp > 3:
        parts.append(
            f"Có {lp} khoảng dừng dài (>2 giây) — dấu hiệu đang tìm từ. "
            f"Hãy luyện tập nói các cụm thay vì từng từ đơn lẻ."
        )
    elif lp > 0:
        parts.append(f"Có {lp} khoảng dừng dài — mức chấp nhận được.")

    # Speaking length
    wc = fluency.get("speaking_length_words", 0)
    if wc < 80:
        parts.append(
            f"Bài nói khá ngắn ({wc} từ). "
            f"IELTS Part 2 nên đạt ít nhất 100-120 từ để thể hiện đủ nội dung."
        )

    # Coherence
    dm_variety = coherence.get("discourse_marker_variety", 0)
    dm_used = coherence.get("discourse_markers_used", {})
    if dm_variety < 0.33:
        parts.append(
            "Sử dụng ít discourse markers. Thêm các từ như 'firstly', 'however', "
            "'furthermore', 'in conclusion' để bài nói mạch lạc hơn."
        )
    else:
        groups = list(dm_used.keys())
        parts.append(
            f"Đã dùng discourse markers từ các nhóm: {', '.join(groups)}. Tốt!"
        )

    return " | ".join(parts) if parts else "Fluency ở mức chấp nhận được."


# ──────────────────────────────────────────────────────────
# LEXICAL RESOURCE FEEDBACK
# ──────────────────────────────────────────────────────────

def _fb_lexical(lexical: dict, score: float) -> str:
    parts = []

    mattr = lexical.get("mattr", 0)
    if mattr < 0.40:
        parts.append(
            f"Lexical diversity thấp (MATTR = {mattr:.2f}). "
            f"Bạn đang lặp lại nhiều từ. Hãy cố gắng dùng từ đồng nghĩa."
        )
    elif mattr > 0.55:
        parts.append(f"Lexical diversity tốt (MATTR = {mattr:.2f}).")

    # Từ lặp
    rep_word = lexical.get("most_repeated_word", "")
    rep_count = lexical.get("max_word_repetition", 0)
    total = max(1, lexical.get("total_words", 1))
    if rep_word and rep_count / total > 0.06:
        parts.append(
            f"Từ '{rep_word}' xuất hiện {rep_count} lần "
            f"({rep_count/total*100:.1f}% tổng số từ). "
            f"Hãy thay bằng từ đồng nghĩa để tăng điểm Lexical Resource."
        )

    # AWL
    awl = lexical.get("advanced_vocab_ratio", 0)
    if awl > 0.08:
        parts.append(
            f"Sử dụng nhiều từ học thuật (AWL ratio = {awl:.1%}) — rất tốt!"
        )
    elif awl < 0.04:
        parts.append(
            f"Nên dùng thêm từ học thuật và ít phổ biến hơn "
            f"(AWL ratio hiện tại: {awl:.1%})."
        )

    return " | ".join(parts) if parts else "Vốn từ ở mức chấp nhận được."


# ──────────────────────────────────────────────────────────
# GRAMMAR FEEDBACK
# ──────────────────────────────────────────────────────────

def _fb_grammar(grammar: dict, score: float) -> str:
    parts = []

    err_rate = grammar.get("grammar_error_rate", 0)
    err_count = grammar.get("grammar_error_count", 0)
    errors_detail = grammar.get("grammar_errors_detail", [])

    if err_rate > 6:
        parts.append(
            f"Phát hiện {err_count} lỗi ngữ pháp ({err_rate:.1f} lỗi/100 từ). "
            f"Cần cải thiện đáng kể độ chính xác ngữ pháp."
        )
    elif err_rate > 3:
        parts.append(
            f"Phát hiện {err_count} lỗi ngữ pháp ({err_rate:.1f} lỗi/100 từ). "
            f"Ở mức trung bình — cần chú ý hơn."
        )
    elif err_count > 0:
        parts.append(
            f"Chỉ có {err_count} lỗi ngữ pháp nhỏ — khá tốt."
        )
    else:
        parts.append("Không phát hiện lỗi ngữ pháp đáng kể — xuất sắc!")

    # Ví dụ lỗi cụ thể (tối đa 2)
    if errors_detail:
        examples = []
        for err in errors_detail[:2]:
            ctx = err.get("context", "").strip()
            msg = err.get("message", "")
            sug = err.get("suggestions", [])
            if ctx and msg:
                ex = f'"{ctx}" → {msg}'
                if sug:
                    ex += f' (gợi ý: {sug[0]})'
                examples.append(ex)
        if examples:
            parts.append("Ví dụ lỗi: " + "; ".join(examples))

    # Complex clauses
    cc_ratio = grammar.get("complex_clause_ratio", 0)
    if cc_ratio < 0.20:
        parts.append(
            "Câu đơn chiếm phần lớn. Hãy dùng thêm mệnh đề phụ: "
            "'which', 'because', 'although', 'when', 'if'..."
        )
    elif cc_ratio > 0.40:
        parts.append(
            f"Câu phức đa dạng tốt (complex clause ratio = {cc_ratio:.0%})."
        )

    return " | ".join(parts) if parts else "Ngữ pháp ở mức chấp nhận được."


# ──────────────────────────────────────────────────────────
# PRONUNCIATION FEEDBACK
# ──────────────────────────────────────────────────────────

def _fb_pronunciation(sub_scores: dict, score: float) -> str:
    """
    Pronunciation feedback dựa trên sub-score và acoustic features.
    """
    if score >= 8.0:
        return (
            "Phát âm rất tốt — ngữ điệu phong phú, giọng rõ ràng. "
            "Tiếp tục duy trì nhịp điệu tự nhiên."
        )
    elif score >= 7.0:
        return (
            "Phát âm tốt. Hãy chú ý thêm vào intonation (lên giọng/xuống giọng) "
            "để bài nói nghe tự nhiên hơn."
        )
    elif score >= 6.0:
        return (
            "Phát âm ở mức trung bình. Tập trung vào: "
            "(1) Word stress — nhấn đúng âm tiết. "
            "(2) Connected speech — nối các từ khi nói tự nhiên."
        )
    else:
        return (
            "Phát âm cần cải thiện nhiều. Hãy: "
            "(1) Luyện nghe và bắt chước người bản ngữ. "
            "(2) Ghi âm và so sánh với mẫu chuẩn. "
            "(3) Chú ý final consonants — đừng nuốt âm cuối."
        )


# ──────────────────────────────────────────────────────────
# STRENGTHS & IMPROVEMENTS
# ──────────────────────────────────────────────────────────

def _get_strengths(fluency, lexical, grammar, coherence, sub_scores) -> list:
    strengths = []

    if sub_scores.get("fluency_coherence", 0) >= 7.0:
        strengths.append("Fluency tốt — nói trôi chảy, ít ngắt nghỉ không cần thiết.")

    if fluency.get("filler_count", 99) <= 2:
        strengths.append("Rất ít filler words — bài nói nghe tự nhiên.")

    dm_variety = coherence.get("discourse_marker_variety", 0)
    if dm_variety >= 0.50:
        strengths.append(
            f"Sử dụng discourse markers đa dạng từ {int(dm_variety*6)} nhóm chức năng."
        )

    if lexical.get("mattr", 0) >= 0.55:
        strengths.append(f"Vốn từ phong phú (MATTR = {lexical.get('mattr', 0):.2f}).")

    if grammar.get("complex_clause_ratio", 0) >= 0.40:
        strengths.append("Sử dụng câu phức đa dạng — thể hiện Grammatical Range tốt.")

    if grammar.get("grammar_error_rate", 99) <= 2.0:
        strengths.append("Độ chính xác ngữ pháp cao.")

    if sub_scores.get("pronunciation", 0) >= 7.5:
        strengths.append("Phát âm và ngữ điệu rõ ràng, tự nhiên.")

    return strengths if strengths else ["Hoàn thành bài nói — tiếp tục luyện tập!"]


def _get_improvements(fluency, lexical, grammar, coherence, sub_scores) -> list:
    improvements = []

    fc = fluency.get("filler_count", 0)
    if fc > 5:
        improvements.append(f"Giảm filler words (hiện tại: {fc} lần).")

    lp = fluency.get("long_pause_count", 0)
    if lp > 3:
        improvements.append(f"Giảm khoảng dừng dài ({lp} lần > 2 giây).")

    wc = fluency.get("speaking_length_words", 0)
    if wc < 100:
        improvements.append(
            f"Tăng độ dài bài nói (hiện tại: {wc} từ — Band 7+ cần ≥ 100 từ cho Part 2)."
        )

    rep_word = lexical.get("most_repeated_word", "")
    rep_count = lexical.get("max_word_repetition", 0)
    total = max(1, lexical.get("total_words", 1))
    if rep_word and rep_count / total > 0.06:
        improvements.append(
            f"Đa dạng hóa từ vựng — '{rep_word}' xuất hiện {rep_count} lần."
        )

    if grammar.get("grammar_error_rate", 0) > 5:
        improvements.append(
            f"Cải thiện ngữ pháp "
            f"({grammar.get('grammar_error_count', 0)} lỗi phát hiện)."
        )

    if grammar.get("complex_clause_ratio", 0) < 0.20:
        improvements.append("Dùng thêm câu phức và mệnh đề phụ.")

    if coherence.get("discourse_marker_variety", 0) < 0.33:
        improvements.append(
            "Thêm discourse markers: 'furthermore', 'however', 'as a result'..."
        )

    if sub_scores.get("pronunciation", 0) < 6.5:
        improvements.append("Luyện tập intonation và word stress.")

    return improvements if improvements else ["Tiếp tục duy trì chất lượng hiện tại!"]
