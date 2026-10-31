"""
coherence.py
Đo Coherence & Cohesion từ transcript:
  - Discourse markers: firstly, however, furthermore, ...
  - Semantic coherence: cosine similarity giữa các câu liền kề
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Lazy-load model ───────────────────────────────────────
_sbert_model = None

def _get_sbert():
    global _sbert_model
    if _sbert_model is None:
        # all-MiniLM-L6-v2: nhẹ (80MB), đủ tốt, chạy local
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


# ─── Discourse Markers theo nhóm chức năng ────────────────
DISCOURSE_MARKERS = {
    "sequence":    ["firstly", "secondly", "thirdly", "finally",
                    "to begin with", "to start with", "first of all",
                    "next", "then", "lastly"],
    "contrast":    ["however", "on the other hand", "although", "though",
                    "nevertheless", "despite", "in contrast",
                    "while", "whereas", "yet"],
    "addition":    ["furthermore", "moreover", "in addition", "besides",
                    "additionally", "also", "as well as", "not only"],
    "cause_effect":["therefore", "thus", "consequently", "as a result",
                    "because", "since", "due to", "hence"],
    "example":     ["for example", "for instance", "such as",
                    "to illustrate", "namely", "like"],
    "conclusion":  ["in conclusion", "to sum up", "to conclude",
                    "overall", "in summary", "to summarise"],
}

ALL_MARKERS = {m for markers in DISCOURSE_MARKERS.values() for m in markers}


def extract_coherence_features(transcript: str) -> dict:
    """
    Tính các đặc trưng Coherence & Cohesion.

    Returns:
        {
            "discourse_marker_count": int,
            "discourse_marker_variety": float,  # số nhóm có dùng / 6
            "discourse_markers_used": dict,      # {group: [markers used]}
            "semantic_coherence_score": float,   # cosine sim TB giữa câu liền kề
        }
    """
    if not transcript.strip():
        return _empty_coherence()

    text_lower = transcript.lower()

    # 1. Detect discourse markers
    markers_used = {}
    total_marker_count = 0
    groups_used = 0

    for group, markers in DISCOURSE_MARKERS.items():
        found = []
        for marker in markers:
            # Dùng word boundary để tránh false positive
            pattern = r'\b' + re.escape(marker) + r'\b'
            if re.search(pattern, text_lower):
                found.append(marker)
                total_marker_count += len(re.findall(pattern, text_lower))
        if found:
            markers_used[group] = found
            groups_used += 1

    discourse_marker_variety = round(groups_used / len(DISCOURSE_MARKERS), 3)

    # 2. Semantic coherence giữa các câu liền kề
    sentences = _split_sentences(transcript)
    if len(sentences) >= 2:
        semantic_score = _compute_semantic_coherence(sentences)
    else:
        semantic_score = 0.5  # Không đủ câu để đo

    return {
        "discourse_marker_count": total_marker_count,
        "discourse_marker_variety": discourse_marker_variety,
        "discourse_markers_used": markers_used,
        "semantic_coherence_score": semantic_score,
    }


def _compute_semantic_coherence(sentences: list) -> float:
    """
    Tính trung bình cosine similarity giữa các câu liền kề.
    Score cao → bài nói mạch lạc, các ý liên kết với nhau.
    """
    try:
        model = _get_sbert()
        embeddings = model.encode(sentences, show_progress_bar=False)
        scores = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            scores.append(float(sim))
        return round(float(np.mean(scores)), 3) if scores else 0.5
    except Exception:
        return 0.5


def _split_sentences(text: str) -> list:
    """Tách văn bản thành các câu."""
    parts = re.split(r'[.!?]+', text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def _empty_coherence() -> dict:
    return {
        "discourse_marker_count": 0,
        "discourse_marker_variety": 0.0,
        "discourse_markers_used": {},
        "semantic_coherence_score": 0.0,
    }
