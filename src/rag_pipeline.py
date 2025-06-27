import streamlit as st
import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import os
import logging
from typing import List, Tuple, Optional, Dict
from src.config import Config
from src.utils import parse_punishment_years

logger = logging.getLogger(__name__)

def get_embeddings(texts: List[str], model_name: str) -> Tuple[np.ndarray, int]:
    """
    Generates embeddings for a list of texts using a pre-trained SBERT model.
    
    Args:
        texts: List of text strings to embed.
        model_name: Name of the SentenceTransformer model.
    
    Returns:
        Tuple of embeddings (numpy array) and embedding dimension.
    """
    try:
        model = SentenceTransformer(model_name)
        with st.spinner("Generating embeddings..."):
            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings, model.get_sentence_embedding_dimension()
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def build_and_save_faiss_index(embeddings: np.ndarray, texts: List[str], embedding_dimension: int, index_path: str, metadata_path: str) -> faiss.IndexFlatL2:
    """
    Builds and saves a FAISS index along with metadata.
    
    Args:
        embeddings: Numpy array of embeddings.
        texts: List of original texts.
        embedding_dimension: Dimension of the embeddings.
        index_path: Path to save the FAISS index.
        metadata_path: Path to save the metadata.
    
    Returns:
        FAISS index object.
    """
    try:
        index = faiss.IndexFlatL2(embedding_dimension)
        index.add(embeddings)
        faiss.write_index(index, index_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved FAISS index to {index_path} and metadata to {metadata_path}")
        return index
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        raise

def load_faiss_index_and_metadata(index_path: str, metadata_path: str) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Loads a FAISS index and its corresponding metadata.
    
    Args:
        index_path: Path to the FAISS index file.
        metadata_path: Path to the metadata file.
    
    Returns:
        Tuple of FAISS index and metadata list.
    """
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"Loaded FAISS index from {index_path} and metadata from {metadata_path}")
        return index, metadata
    except FileNotFoundError as e:
        logger.error(f"FAISS index or metadata not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading FAISS index or metadata: {e}")
        raise

@st.cache_resource
def initialize_rag_components():
    """Initializes RAG components (data, index, models)."""
    from src.data_processing import load_data_structured
    structured_ipc_data = load_data_structured(Config.DATA_FILE)
    ipc_data_for_embeddings = [item["original_text"] for item in structured_ipc_data]

    if not os.path.exists(Config.VECTOR_DB_PATH) or not os.path.exists(Config.METADATA_PATH):
        ipc_embeddings, embedding_dim = get_embeddings(ipc_data_for_embeddings, Config.EMBEDDING_MODEL_NAME)
        ipc_faiss_index = build_and_save_faiss_index(
            ipc_embeddings, ipc_data_for_embeddings, embedding_dim, Config.VECTOR_DB_PATH, Config.METADATA_PATH
        )
        ipc_metadata = ipc_data_for_embeddings
    else:
        ipc_faiss_index, ipc_metadata = load_faiss_index_and_metadata(Config.VECTOR_DB_PATH, Config.METADATA_PATH)
        
    retrieval_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
    re_ranker_model = CrossEncoder(Config.RE_RANKER_MODEL_NAME)
    
    return structured_ipc_data, ipc_faiss_index, ipc_metadata, retrieval_model, re_ranker_model

def retrieve_relevant_ipc_hybrid(
    query: str,
    structured_ipc_data: List[Dict],
    faiss_index: faiss.IndexFlatL2,
    metadata: List[str],
    embedding_model: SentenceTransformer,
    re_ranker_model: CrossEncoder,
    k_initial: int,
    k_final: int
) -> List[str]:
    """
    Retrieves relevant IPC sections using a hybrid approach.
    
    Args:
        query: User query string.
        signed_ipc_data: List of structured IPC data.
        faiss_index: FAISS index for semantic search.
        metadata: Metadata associated with FAISS index.
        embedding_model: SentenceTransformer model for embeddings.
        re_ranker_model: CrossEncoder model for re-ranking.
        k_initial: Number of initial documents to retrieve.
        k_final: Number of final documents after re-ranking.
    
    Returns:
        List of relevant IPC section texts.
    """
    import re
    min_years_threshold = None
    punishment_years_filter_match = re.search(r'greater than (\d+)\s*(?:years|year)?', query, re.IGNORECASE)
    if punishment_years_filter_match:
        min_years_threshold = int(punishment_years_filter_match.group(1))
        st.info(f"Applying numerical filter: Punishment greater than {min_years_threshold} years.")
        filtered_by_condition = [
            item["original_text"] for item in structured_ipc_data
            if item["punishment_years"] is not None and item["punishment_years"] > min_years_threshold
        ]
        if filtered_by_condition:
            sentence_pairs = [[query, doc] for doc in filtered_by_condition]
            if sentence_pairs:
                rerank_scores = re_ranker_model.predict(sentence_pairs)
                scored_documents = sorted(list(zip(filtered_by_condition, rerank_scores)), key=lambda x: x[1], reverse=True)
                return [doc for doc, score in scored_documents[:20]]
            return []
        st.warning("Numerical filter applied but found no matching sections. Falling back to general search.")

    retrieved_sections = []
    query_lower = query.lower()
    
    # Keyword matching
    keyword_matches = []
    if 'child' in query_lower or 'minor' in query_lower:
        for item in structured_ipc_data:
            if 'child' in item["original_text"].lower() or 'minor' in item["original_text"].lower():
                keyword_matches.append(item["original_text"])

    # Exact IPC section match
    match = re.search(r'IPC\s*(\d+[A-Z]*)', query, re.IGNORECASE)
    if match:
        section_id_query = f"IPC {match.group(1).upper()}"
        for section_text in metadata:
            if section_text.lower().startswith(section_id_query.lower()):
                retrieved_sections.append(section_text)
                break
    
    # Semantic search
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k_initial)
    
    semantic_matches = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadata):
            semantic_matches.append(metadata[idx])
            
    combined_results_for_reranking = []
    seen = set()
    for item in keyword_matches + retrieved_sections + semantic_matches:
        if item not in seen:
            combined_results_for_reranking.append(item)
            seen.add(item)

    if not combined_results_for_reranking:
        return []

    sentence_pairs = [[query, doc] for doc in combined_results_for_reranking]
    rerank_scores = re_ranker_model.predict(sentence_pairs)
    scored_documents = sorted(list(zip(combined_results_for_reranking, rerank_scores)), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_documents[:k_final]]

def generate_answer_with_ollama(query: str, retrieved_ipc_sections: List[str], ollama_model: str) -> str:
    """
    Generates an answer using Ollama LLM based on retrieved IPC sections.
    
    Args:
        query: User query string.
        retrieved_ipc_sections: List of relevant IPC section texts.
        ollama_model: Name of the Ollama model.
    
    Returns:
        Generated response string.
    """
    if not retrieved_ipc_sections:
        return "Mujhe aapki query ke liye apne database mein koi relevant IPC section nahi mila. Kya aap kripya ise doosre tarike se poochh sakte hain ya aur details de sakte hain?"

    context = "\n".join(retrieved_ipc_sections)
    prompt = f"""Aap Indian Penal Code (IPC) mein expert ek helpful aur jaankari dene wale legal assistant hain.
    Aapka maksad IPC ke concepts aur saza ko **bahut hi saral aur aasaan bhasha mein samjhana** hai, jaise aap ek 10 saal ke bachche ko samjha rahe hon.
    Jahan tak ho sake, complex legal shabdon se bachen, ya agar zaroori ho to unhe संक्षेप mein samjhayein.

    **Aapko sirf neeche diye gaye IPC sections ke aadhar par hi jawab dena hai.**
    **Aapki sabse pehli priority hai ki har us IPC section aur uski saza ko list karein jo user ke criteria ko pura karta hai.** Aapko context mein diye gaye kisi bhi relevant section ko chhodna nahi hai.
    Har section ke liye, uska number, naam (title), aur saza ko saaf-saaf batayein.
    **Jawab ko thoda vistar (elaborate) se dein, har section ko thoda aur khol kar samjhayein, taaki user ko behtar samajh aaye.**
    Agar poori tarah se jawab dene ke liye zaroori jaankari diye gaye sections mein nahi hai, to saaf-saaf batayein ki aapke paas di gayi jaankari ke aadhar par vishesh vivaran nahi hai. Apni taraf se koi jaankari na banayein.

    IPC Sections:
    {context}

    User Query: {query}

    Answer:
    """
    try:
        full_response_content = ""
        response_placeholder = st.empty()
        stream = ollama.chat(
            model=ollama_model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3},
            stream=True
        )
        for chunk in stream:
            if 'content' in chunk['message']:
                full_response_content += chunk['message']['content']
                response_placeholder.markdown(full_response_content)
        return full_response_content
    except ollama.ResponseError as e:
        logger.error(f"Ollama error: {e}")
        if "connection refused" in str(e).lower():
            return "Maaf kijiye, main Ollama se connect nahi kar pa raha. Kripya confirm karein ki Ollama chal raha hai (`ollama serve`) aur model download ho gaya hai (`ollama run llama3.1`)."
        elif "model 'llama3.1' not found" in str(e).lower():
            return f"Maaf kijiye, '{ollama_model}' model nahi mila. Kripya confirm karein ki yeh Ollama mein download ho gaya hai (`ollama run {ollama_model}`)."
        return f"Ollama mein ek error aa gayi: {e}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Ek anapekshit error aa gayi: {e}"

