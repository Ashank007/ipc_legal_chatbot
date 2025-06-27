from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    """Configuration class for IPC Legal Chatbot."""
    DATA_FILE = os.getenv("DATA_FILE", "data/legal_data.jsonl")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    RE_RANKER_MODEL_NAME = os.getenv("RE_RANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/ipc_faiss.index")
    METADATA_PATH = os.getenv("METADATA_PATH", "data/ipc_metadata.json")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
    K_INITIAL_RETRIEVAL = int(os.getenv("K_INITIAL_RETRIEVAL", 20))
    K_FINAL_RE_RANKED = int(os.getenv("K_FINAL_RE_RANKED", 15))
