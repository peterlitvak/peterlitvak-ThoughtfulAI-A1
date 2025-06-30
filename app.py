"""
Streamlit UI for the Thoughtful AI mini-RAG chatbot.
- Loads system prompt from a markdown file.
- Uses RAGService for retrieval-augmented chat with streaming.
- Maintains chat history in Streamlit session state.
"""
import streamlit as st
import os
from rag_backend import RAGService, InMemoryChatHistoryProvider
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load system prompt from markdown file
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.md")
try:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except (FileNotFoundError, OSError):
    st.error(f"System prompt file not found: {PROMPT_PATH}")
    st.stop()

# Path to the knowledge base file
KB_PATH = os.path.join(os.path.dirname(__file__), "data.json")

# Initialize Streamlit session state for chat session and history
if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(8).hex()
if "history" not in st.session_state:
    st.session_state.history = []

# Cache the RAGService instance for performance
@st.cache_resource(show_spinner=False)
def get_rag_service():
    """
    Instantiate and return a RAGService using the system prompt and knowledge base.
    Returns
    -------
    RAGService
        The retrieval-augmented generation service instance.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found in environment. Please set it in your .env file.")
        st.stop()
    try:
        return RAGService(
            system_prompt=SYSTEM_PROMPT,
            knowledge_base_path=KB_PATH,
            chat_memory_provider=InMemoryChatHistoryProvider(),
            llm_api_key=api_key,
            llm=None
        )
    except ValueError as e:
        st.error(f"Failed to initialize RAGService: {e}")
        st.stop()
rag = get_rag_service()

st.set_page_config(page_title="Thoughtful AI Chatbot", page_icon="💬", layout="centered")
st.title("💬 Thoughtful AI mini-RAG")
st.markdown("""
Ask a question about Thoughtful AI.
""")

# Display chat history from session state
for user_msg, ai_msg in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(ai_msg)

# User input box at the bottom
user_input = st.chat_input("Type your question and press Enter...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        response_area = st.empty()
        full_response = ""
        try:
            # Stream the answer from the RAG service as chunks arrive
            for chunk in rag.chat(user_input, session_id=st.session_state.session_id, stream=True):
                full_response += chunk
                response_area.markdown(full_response)
        except Exception as e:
            st.error(f"Error while generating answer: {e}")
            st.stop()
        # Add the turn to chat history in session state
        st.session_state.history.append((user_input, full_response))
