"""
evaluate.py — Entry Point
Nhận vào file .mp3 → chạy toàn bộ pipeline → xuất JSON kết quả

Sử dụng:
    python evaluate.py "path/to/audio.mp3"
    python evaluate.py "path/to/audio.mp3" --model small
    python evaluate.py "path/to/audio.mp3" --output result.json
"""

import json
import sys
import argparse
from pathlib import Path

# Import các modules
from utils.transcriber    import transcribe
from utils.audio_features import extract_fluency_features, extract_pronunciation_features
from utils.text_features  import extract_lexical_features, extract_grammar_features
from utils.coherence      import extract_coherence_features
from utils.scorer         import (score_fluency_coherence, score_lexical_resource,
                                   score_grammatical_range_accuracy, score_pronunciation)
from utils.calibration    import calibrate, round_to_ielts_band
from utils.calibration    import calibrate, round_to_ielts_band
from utils.feedback       import generate_feedback
from utils.topic_relevance import check_topic_relevance


def evaluate(audio_path: str, question: str, threshold: float, model_size: str = "base") -> dict:
    """
    Pipeline chính: nhận file audio → trả JSON kết quả đầy đủ.

    Args:
        audio_path: Đường dẫn file .mp3 / .wav
        question: Câu hỏi IELTS Speaking
        threshold: Ngưỡng topic similarity
        model_size: Whisper model size (tiny/base/small/medium)

    Returns:
        dict kết quả đầy đủ — sẵn sàng để bọc API
    """
    print(f"\n[1/6] Transcribing: {audio_path}")
    trans = transcribe(audio_path, model_size=model_size)

    print(f"\n[+] Checking Topic Relevance...")
    relevance_result = check_topic_relevance(question, trans["transcript"], threshold)
    
    print("\n" + "="*40)
    print("TOPIC RELEVANCE ANALYSIS")
    print("="*40)
    print(f"\nQuestion:\n{question}")
    print(f"\nSimilarity Score : {relevance_result['topic_similarity']:.2f}")
    print(f"Threshold        : {relevance_result['threshold']:.2f}")
    
    if not relevance_result["continue_scoring"]:
        print("\nResult:\n❌ OFF TOPIC\n")
        print("IELTS Speaking scoring has been skipped.")
        return relevance_result
        
    print("\nResult:\n✅ ON TOPIC\n")
    print("Proceeding to IELTS Speaking Scoring...\n")

    print(f"[2/6] Extracting audio features...")
    fluency_feats = extract_fluency_features(
        words=trans["words"],
        duration_seconds=trans["duration_seconds"],
        filler_count=trans["filler_count"],
        audio_path=audio_path
    )
    # Merge filler_words từ transcriber vào fluency_feats
    fluency_feats["filler_words"] = trans["filler_words"]

    pron_feats = extract_pronunciation_features(audio_path)

    print(f"[3/6] Extracting text features...")
    lexical_feats  = extract_lexical_features(trans["transcript"])
    grammar_feats  = extract_grammar_features(trans["transcript"])

    print(f"[4/6] Computing coherence...")
    coherence_feats = extract_coherence_features(trans["transcript"])

    print(f"[5/6] Scoring (Rule-based Expert System)...")
    sub_scores = {
        "fluency_coherence":           score_fluency_coherence(fluency_feats, coherence_feats),
        "lexical_resource":            score_lexical_resource(lexical_feats),
        "grammatical_range_accuracy":  score_grammatical_range_accuracy(grammar_feats),
        "pronunciation":               score_pronunciation(pron_feats, fluency_feats),
    }

    # Round sub-scores về thang 0.5
    sub_scores_rounded = {k: round_to_ielts_band(v) for k, v in sub_scores.items()}

    # Calibration → overall band
    overall_band = calibrate(sub_scores)

    print(f"[6/6] Generating feedback...")
    feedback = generate_feedback(
        fluency=fluency_feats,
        lexical=lexical_feats,
        grammar=grammar_feats,
        coherence=coherence_feats,
        sub_scores=sub_scores_rounded,
    )

    # ─── Build output JSON ───────────────────────────────
    result = {
        # Meta
        "audio_file": str(audio_path),
        "duration_seconds": trans["duration_seconds"],

        # Transcript
        "transcript": trans["transcript"],

        # Scores
        "overall_band": overall_band,
        "sub_scores": sub_scores_rounded,
        "raw_sub_scores": {k: round(v, 2) for k, v in sub_scores.items()},

        # Raw features (để debug / hiệu chỉnh calibration)
        "raw_features": {
            # Fluency
            "speech_rate_wpm":     fluency_feats.get("speech_rate_wpm"),
            "articulation_rate":   fluency_feats.get("articulation_rate"),
            "long_pause_count":    fluency_feats.get("long_pause_count"),
            "short_pause_count":   fluency_feats.get("short_pause_count"),
            "silent_ratio":        fluency_feats.get("silent_ratio"),
            "speaking_length_words": fluency_feats.get("speaking_length_words"),
            "filler_words":        fluency_feats.get("filler_words"),
            "filler_count":        fluency_feats.get("filler_count"),
            # Pronunciation
            "pitch_mean":          pron_feats.get("pitch_mean"),
            "pitch_std":           pron_feats.get("pitch_std"),
            "jitter_local":        pron_feats.get("jitter_local"),
            "shimmer_local":       pron_feats.get("shimmer_local"),
            "hnr":                 pron_feats.get("hnr"),
            "rhythm_score":        pron_feats.get("rhythm_score"),
            # Lexical
            "mattr":               lexical_feats.get("mattr"),
            "advanced_vocab_ratio": lexical_feats.get("advanced_vocab_ratio"),
            "rare_word_ratio":     lexical_feats.get("rare_word_ratio"),
            "most_repeated_word":  lexical_feats.get("most_repeated_word"),
            "max_word_repetition": lexical_feats.get("max_word_repetition"),
            # Grammar
            "grammar_error_count": grammar_feats.get("grammar_error_count"),
            "grammar_error_rate":  grammar_feats.get("grammar_error_rate"),
            "complex_clause_ratio": grammar_feats.get("complex_clause_ratio"),
            "tense_consistent":    grammar_feats.get("tense_consistent"),
            # Coherence
            "discourse_marker_count":   coherence_feats.get("discourse_marker_count"),
            "discourse_marker_variety": coherence_feats.get("discourse_marker_variety"),
            "semantic_coherence_score": coherence_feats.get("semantic_coherence_score"),
            # Self-correction
            "self_correction_count": trans.get("self_correction_count", 0),
            "self_corrections":      trans.get("self_corrections", []),
        },

        # Feedback
        "feedback": feedback,
        "topic_relevance": relevance_result,
    }

    return result


# ─── CLI Interface ────────────────────────────────────────

import tkinter as tk
from tkinter import filedialog

def main():
    parser = argparse.ArgumentParser(
        description="IELTS Speaking Scorer — Rule-based Expert System"
    )
    parser.add_argument("audio", nargs="?", default=None, help="Đường dẫn file audio (.mp3 / .wav)")
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (mặc định: base)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Lưu kết quả ra file JSON (tùy chọn)"
    )
    args = parser.parse_args()

    # Load config.json for threshold
    config_path = Path("config.json")
    default_threshold = 0.40
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                default_threshold = config.get("topic_similarity_threshold", 0.40)
        except Exception:
            pass

    # Step 1: Nhập câu hỏi
    print("\nEnter the IELTS Speaking Question.")
    question = input("> ").strip()
    if not question:
        print("[WARNING] Question cannot be empty. Please restart.")
        sys.exit(1)

    # Step 2: Nhập threshold
    print(f"\nCurrent Topic Similarity Threshold: {default_threshold:.2f}")
    print("Press Enter to keep default\nor enter a new value:\n")
    user_thresh = input("> ").strip()
    if user_thresh:
        try:
            threshold = float(user_thresh)
        except ValueError:
            print(f"[ERROR] Invalid threshold. Using default: {default_threshold:.2f}")
            threshold = default_threshold
    else:
        threshold = default_threshold

    audio_path = args.audio

    if not audio_path:
        # Hiện hộp thoại chọn file nếu không truyền qua Command Line
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True) # Hiện hộp thoại lên trên cùng
        audio_path = filedialog.askopenfilename(
            title="Chọn file Audio IELTS Speaking",
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a"), ("All Files", "*.*")]
        )
        if not audio_path:
            print("[INFO] Đã hủy chọn file. Thoát chương trình.")
            sys.exit(0)

    if not Path(audio_path).exists():
        print(f"[ERROR] Không tìm thấy file: {audio_path}")
        sys.exit(1)

    result = evaluate(audio_path, question=question, threshold=threshold, model_size=args.model)

    if result.get("status") == "off_topic":
        # Handle early termination cleanly
        if args.output:
            out_path = args.output
        else:
            stem = Path(audio_path).stem
            json_dir = Path(r"C:\Users\thgam\OneDrive\Documents\Audio test ielts\Json")
            json_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(json_dir / f"{stem}_scored.json")
            
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Kết quả đã lưu: {out_path}")
        return

    # In kết quả ra terminal
    print("\n" + "="*60)
    print("  IELTS SPEAKING SCORER — KẾT QUẢ")
    print("="*60)
    print(f"  Overall Band : {result['overall_band']}")
    print(f"  Fluency & Coherence       : {result['sub_scores']['fluency_coherence']}")
    print(f"  Lexical Resource          : {result['sub_scores']['lexical_resource']}")
    print(f"  Grammatical Range & Acc.  : {result['sub_scores']['grammatical_range_accuracy']}")
    print(f"  Pronunciation             : {result['sub_scores']['pronunciation']}")
    print("="*60)
    print(f"\nTranscript:\n  {result['transcript'][:200]}...")
    print(f"\nStrengths:")
    for s in result["feedback"]["strengths"]:
        print(f"  ✅ {s}")
    print(f"\nImprovements:")
    for i in result["feedback"]["improvements"]:
        print(f"  ⚠️  {i}")
    print("="*60)

    # Lưu JSON
    if args.output:
        out_path = args.output
    else:
        stem = Path(audio_path).stem
        # Thư mục đích chứa các file JSON
        json_dir = Path(r"C:\Users\thgam\OneDrive\Documents\Audio test ielts\Json")
        json_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(json_dir / f"{stem}_scored.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Kết quả đã lưu: {out_path}")


if __name__ == "__main__":
    main()
