from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .dataset import DATASET

model = SentenceTransformer(
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    device="cpu"
)

questions = [item["question"] for item in DATASET]
question_embeddings = model.encode(
    questions,
    show_progress_bar=False,
    convert_to_numpy=True
)

def find_best_answer(user_text: str, threshold: float = 0.55):
    user_vec = model.encode(
        [user_text],
        show_progress_bar=False,
        convert_to_numpy=True
    )
    sims = cosine_similarity(user_vec, question_embeddings)[0]
    best_idx = int(np.argmax(sims))

    if sims[best_idx] < threshold:
        return None

    return DATASET[best_idx]["answer"]
