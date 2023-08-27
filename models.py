from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL_PATH, DEVICE, GPU_ID


def get_embedding_model() -> SentenceTransformer: 
    return SentenceTransformer(EMBED_MODEL_PATH, device=f"{DEVICE}:{GPU_ID}")

EMBEDDING_MODEL = get_embedding_model()