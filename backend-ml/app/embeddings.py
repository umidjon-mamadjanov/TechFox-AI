from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .dataset import DATASET

model = SentenceTransformer("all-MiniLM-L6-v2")

questions = [item["question"] for item in DATASET]
question_embeddings = model.encode(questions)

def find_best_answer(user_text: str, threshold: float = 0.6):
    user_vec = model.encode([user_text])
    sims = cosine_similarity(user_vec, question_embeddings)[0]

    best_idx = int(np.argmax(sims))
    best_score = sims[best_idx]

    if best_score < threshold:
        return None

    return DATASET[best_idx]["answer"]
