import streamlit as st
import logging
import logging.config
import yaml
from src.config import Config
from src.data_processing import load_data_structured
from src.rag_pipeline import initialize_rag_components, retrieve_relevant_ipc_hybrid, generate_answer_with_ollama

# Set up logging
with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f)
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the Streamlit IPC Legal Chatbot."""
    st.set_page_config(page_title="IPC Legal Chatbot", page_icon="⚖️", layout="wide")
    
    st.title("⚖️ IPC Legal Chatbot - Simple Legal Explanations")
    st.markdown("""
    Welcome! I'm your AI assistant for Indian Penal Code (IPC) sections. 
    Just ask me any question about IPC, and I'll explain it in **simple, easy-to-understand language**.
    """)

    # Load components with caching
    try:
        structured_ipc_data, ipc_faiss_index, ipc_metadata, retrieval_model, re_ranker_model = initialize_rag_components()
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        st.error(f"Error initializing application: {e}")
        return

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_query := st.chat_input("Ask a question about IPC..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Finding relevant IPC sections and generating response..."):
                try:
                    retrieved_sections = retrieve_relevant_ipc_hybrid(
                        user_query,
                        structured_ipc_data,
                        ipc_faiss_index,
                        ipc_metadata,
                        retrieval_model,
                        re_ranker_model,
                        Config.K_INITIAL_RETRIEVAL,
                        Config.K_FINAL_RE_RANKED
                    )
                    full_answer = generate_answer_with_ollama(user_query, retrieved_sections, Config.OLLAMA_MODEL)
                    st.session_state.messages.append({"role": "assistant", "content": full_answer})
                except Exception as e:
                    logger.error(f"Error processing query '{user_query}': {e}")
                    st.error(f"Error processing your query: {e}")

    st.markdown("---")
    st.caption("Powered by Ollama (Llama 3.1), Sentence-Transformers, and FAISS.")

if __name__ == "__main__":
    main()
