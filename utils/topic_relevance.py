from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Use a lightweight, fast, and local model
# We can load the model lazily to avoid loading it when it's not needed.
_model = None

def get_model():
    global _model
    if _model is None:
        print("[INFO] Loading Sentence-Transformer model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def check_topic_relevance(question: str, transcript: str, threshold: float) -> dict:
    """
    Checks if the given transcript is relevant to the question using semantic similarity.
    
    Args:
        question: The IELTS Speaking Question
        transcript: The text transcribed from the user's audio
        threshold: The similarity threshold (e.g. 0.40)
        
    Returns:
        A dictionary with the relevance results.
    """
    model = get_model()
    
    question_embedding = model.encode(question)
    transcript_embedding = model.encode(transcript)
    
    # Cosine similarity expects 2D arrays
    similarity = cosine_similarity(
        [question_embedding],
        [transcript_embedding]
    )[0][0]
    
    similarity_score = float(similarity)
    
    is_on_topic = similarity_score >= threshold
    
    if is_on_topic:
        return {
            "status": "on_topic",
            "topic_similarity": round(similarity_score, 2),
            "threshold": threshold,
            "continue_scoring": True
        }
    else:
        return {
            "status": "off_topic",
            "topic_similarity": round(similarity_score, 2),
            "threshold": threshold,
            "continue_scoring": False,
            "message": "The response is considered off-topic. IELTS Speaking scoring was skipped."
        }
