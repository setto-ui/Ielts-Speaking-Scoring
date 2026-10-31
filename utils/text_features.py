"""
text_features.py
Trích xuất đặc trưng văn bản từ transcript:
  - Lexical Resource: MATTR, AWL, rare words, repetition
  - Grammar: error rate (LanguageTool), clause variety (spaCy), tense consistency
  - Speaking patterns: self-correction, hesitation
"""

import re
import math
import spacy
import language_tool_python
from wordfreq import word_frequency
from collections import Counter

# ─── Khởi tạo models (lazy load) ──────────────────────────
_nlp = None
_grammar_tool = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

def _get_grammar_tool():
    global _grammar_tool
    if _grammar_tool is None:
        _grammar_tool = language_tool_python.LanguageTool("en-US")
    return _grammar_tool


# ─── Academic Word List (AWL) ──────────────────────────────
AWL = {
    "analysis", "approach", "area", "assess", "assume", "authority",
    "available", "benefit", "concept", "consistent", "constitute",
    "context", "contract", "create", "data", "define", "derive",
    "distribute", "economy", "environment", "establish", "estimate",
    "evidence", "export", "factor", "financial", "function", "identify",
    "income", "indicate", "individual", "interpret", "involve", "issue",
    "labour", "legal", "major", "method", "occur", "percent", "period",
    "policy", "principle", "process", "require", "research", "respond",
    "role", "section", "sector", "significant", "similar", "source",
    "structure", "theory", "variable", "volume", "promote", "crucial",
    "innovative", "implement", "impact", "generate", "enhance",
    "utilize", "obtain", "acquire", "achieve", "maintain", "sustain",
    "contribute", "demonstrate", "emphasize", "highlight", "illustrate",
    "indicate", "reveal", "suggest", "support", "facilitate"
}

# Dependency labels cho mệnh đề phụ (complex sentences)
COMPLEX_CLAUSE_DEPS = {"advcl", "relcl", "ccomp", "xcomp", "acl"}


# ─────────────────────────────────────────────────
# LEXICAL RESOURCE FEATURES
# ─────────────────────────────────────────────────

def extract_lexical_features(transcript: str) -> dict:
    """
    Tính các đặc trưng Lexical Resource.
    Dùng MATTR thay TTR để tránh phụ thuộc độ dài bài nói.
    """
    words = _tokenize(transcript)
    if len(words) < 5:
        return _empty_lexical()

    return {
        "mattr": _compute_mattr(words, window_size=50),
        "advanced_vocab_ratio": _compute_awl_ratio(words),
        "rare_word_ratio": _compute_rare_word_ratio(words),
        "most_repeated_word": _get_most_repeated(words),
        "max_word_repetition": _get_max_repetition(words),
        "total_unique_words": len(set(words)),
        "total_words": len(words),
    }


def _compute_mattr(words: list, window_size: int = 50) -> float:
    """
    Moving Average Type-Token Ratio (MATTR).
    Tính TTR trên cửa sổ trượt → không bị ảnh hưởng bởi độ dài bài.
    """
    if len(words) <= window_size:
        return round(len(set(words)) / len(words), 3)
    ratios = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        ratios.append(len(set(window)) / window_size)
    return round(sum(ratios) / len(ratios), 3)


def _compute_awl_ratio(words: list) -> float:
    """Tỷ lệ từ thuộc Academic Word List."""
    awl_count = sum(1 for w in words if w in AWL)
    return round(awl_count / max(1, len(words)), 3)


def _compute_rare_word_ratio(words: list) -> float:
    """
    Tỷ lệ từ ít phổ biến (frequency < 1e-5 trong wordfreq corpus).
    Từ hiếm nhưng chính xác = dấu hiệu Lexical Resource cao.
    """
    rare = sum(1 for w in words
               if len(w) > 3 and word_frequency(w, "en") < 1e-5)
    return round(rare / max(1, len(words)), 3)


def _get_most_repeated(words: list) -> str:
    # Bỏ qua stop words thông dụng khi tìm từ lặp
    STOP = {"the", "a", "an", "is", "are", "was", "were", "i", "you",
            "it", "to", "of", "and", "in", "that", "this", "for", "on"}
    filtered = [w for w in words if w not in STOP and len(w) > 2]
    if not filtered:
        return ""
    return Counter(filtered).most_common(1)[0][0]


def _get_max_repetition(words: list) -> int:
    STOP = {"the", "a", "an", "is", "are", "was", "were", "i", "you",
            "it", "to", "of", "and", "in", "that", "this", "for", "on"}
    filtered = [w for w in words if w not in STOP and len(w) > 2]
    if not filtered:
        return 0
    return Counter(filtered).most_common(1)[0][1]


def _empty_lexical() -> dict:
    return {
        "mattr": 0.0, "advanced_vocab_ratio": 0.0,
        "rare_word_ratio": 0.0, "most_repeated_word": "",
        "max_word_repetition": 0, "total_unique_words": 0, "total_words": 0
    }


# ─────────────────────────────────────────────────
# GRAMMAR FEATURES
# ─────────────────────────────────────────────────

def extract_grammar_features(transcript: str) -> dict:
    """
    Tính các đặc trưng Grammatical Range & Accuracy.
    Kết hợp LanguageTool (accuracy) + spaCy (range/variety).
    """
    if not transcript.strip():
        return _empty_grammar()

    words = _tokenize(transcript)
    word_count = max(1, len(words))

    # LanguageTool: đếm lỗi ngữ pháp
    tool = _get_grammar_tool()
    matches = tool.check(transcript)
    error_count = len(matches)
    error_rate = round((error_count / word_count) * 100, 2)

    # spaCy: phân tích cấu trúc câu
    nlp = _get_nlp()
    doc = nlp(transcript)

    sentences = list(doc.sents)
    num_sentences = max(1, len(sentences))

    # Đếm mệnh đề phụ thực sự (không phải đếm từ > 15)
    complex_clause_count = sum(
        1 for token in doc if token.dep_ in COMPLEX_CLAUSE_DEPS
    )
    complex_clause_ratio = round(complex_clause_count / num_sentences, 3)

    # Độ đa dạng độ dài câu
    sent_lengths = [len(list(s)) for s in sentences]
    sent_length_variance = round(float(
        sum((l - (sum(sent_lengths)/len(sent_lengths)))**2
            for l in sent_lengths) / len(sent_lengths)
    ), 2) if len(sent_lengths) > 1 else 0.0

    # Tense consistency
    tense_consistent, dominant_tense = _check_tense_consistency(doc)

    # Chi tiết lỗi (top 5 để feedback)
    errors_detail = []
    for m in matches[:5]:
        errors_detail.append({
            "message": m.message,
            "context": m.context,
            "suggestions": m.replacements[:3] if m.replacements else []
        })

    return {
        "grammar_error_count": error_count,
        "grammar_error_rate": error_rate,
        "complex_clause_count": complex_clause_count,
        "complex_clause_ratio": complex_clause_ratio,
        "sentence_count": num_sentences,
        "sent_length_variance": sent_length_variance,
        "tense_consistent": tense_consistent,
        "dominant_tense": dominant_tense,
        "grammar_errors_detail": errors_detail,
    }


def _check_tense_consistency(doc) -> tuple:
    """Kiểm tra sự nhất quán về thì trong bài nói."""
    tense_counts = {"past": 0, "present": 0, "future": 0}
    for token in doc:
        if token.pos_ == "VERB":
            if token.tag_ in ("VBD", "VBN"):
                tense_counts["past"] += 1
            elif token.tag_ in ("VBZ", "VBP", "VBG"):
                tense_counts["present"] += 1
            elif token.text.lower() in ("will", "shall", "'ll"):
                tense_counts["future"] += 1
    total = sum(tense_counts.values())
    if total == 0:
        return True, "present"
    dominant = max(tense_counts, key=tense_counts.get)
    ratio = tense_counts[dominant] / total
    return ratio >= 0.6, dominant


def _empty_grammar() -> dict:
    return {
        "grammar_error_count": 0, "grammar_error_rate": 0.0,
        "complex_clause_count": 0, "complex_clause_ratio": 0.0,
        "sentence_count": 0, "sent_length_variance": 0.0,
        "tense_consistent": True, "dominant_tense": "present",
        "grammar_errors_detail": []
    }


# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────

def _tokenize(text: str) -> list:
    return re.findall(r"\b[a-z']+\b", text.lower())
