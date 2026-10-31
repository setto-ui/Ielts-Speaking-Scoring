# IELTS Speaking Scorer

An automated rule-based expert scoring system and evaluation pipeline for IELTS Speaking tests with **Topic Relevance Detection**.

## Features

- **Topic Relevance Detection**: Evaluates whether the candidate's spoken response directly addresses the given IELTS prompt using semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`). Terminates early if the speech is off-topic.
- **Audio Transcription**: Uses `faster-whisper` for fast, accurate local speech-to-text conversion.
- **Fluency & Coherence Analysis**: Measures speech rate (WPM), articulation rate, silent ratios, pause lengths, filler words, discourse markers, and semantic coherence.
- **Pronunciation Assessment**: Analyzes pitch, jitter, shimmer, harmonics-to-noise ratio (HNR), and rhythm metrics via Praat/Parselmouth.
- **Lexical Resource Evaluation**: Assesses MATTR (Moving-Average Type-Token Ratio), advanced vocabulary ratio, rare words, and repetition.
- **Grammatical Range & Accuracy**: Parses syntactic complexity, clause structures, tense consistency, and grammar error rates.
- **Actionable Feedback**: Generates detailed strengths and areas for improvement along with IELTS band sub-scores and overall estimated band.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

## Configuration

Similarity threshold can be configured in `config.json`:
```json
{
    "topic_similarity_threshold": 0.40
}
```

## Usage

Run the evaluation script:
```bash
python evaluate.py
```
Or specify an audio file directly:
```bash
python evaluate.py "path/to/audio.mp3" --model base
```

### Steps during execution:
1. Enter the IELTS Speaking question.
2. Accept or customize the Topic Relevance threshold.
3. Select or provide the audio recording (`.mp3`, `.wav`, etc.).
4. The system transcribes the speech, validates topic relevance, and proceeds to full IELTS band scoring if on-topic.
