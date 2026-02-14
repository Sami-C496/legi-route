"""
LégiRoute — Streamlit Frontend.

Orchestrates user interaction with the RAG system:
- LLM-based intent classification for query routing
- Real-time streaming responses
- Source transparency (expandable citations)
"""

import streamlit as st
import sys
from pathlib import Path

# --- Path Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src.config import settings
    from src.classifier import IntentClassifier, Intent
    from src.retrieval import TrafficRetriever
    from src.generation import TrafficGenerator
except ImportError:
    st.error("❌ Import Error: Please run the app from the project root using 'poetry run streamlit run src/app.py'")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="LégiRoute AI",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stChatInput {border-radius: 15px;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ LégiRoute")
st.caption("Assistant Juridique Intelligent — Code de la Route Français")


# --- System Initialization (Cached) ---
@st.cache_resource
def load_system_components():
    """Initializes all RAG components. Returns None if DB missing."""
    db_path = settings.CHROMA_DB_PATH
    if not db_path.exists():
        return None, None, None

    classifier_instance = IntentClassifier()
    retriever_instance = TrafficRetriever(db_path=str(db_path))
    generator_instance = TrafficGenerator()
    return classifier_instance, retriever_instance, generator_instance


classifier, retriever, generator = load_system_components()

if not retriever:
    st.warning("⚠️ Database not found. Run: `poetry run python src/ingestion/indexing.py`")
    st.stop()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Bonjour ! Je suis LégiRoute. Posez-moi une question sur le Code de la Route (vitesse, alcool, permis...)."
    }]

# --- Render History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Interaction Loop ---
if prompt := st.chat_input("Ex: Quelle est la sanction pour un téléphone au volant ?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        relevant_docs = []

        # --- Intent Classification ---
        intent = classifier.classify(prompt)

        if intent == Intent.OFF_TOPIC:
            full_response = "Je suis spécialisé dans le Code de la Route français. Je ne peux pas répondre à cette question, mais n'hésitez pas à me poser une question sur le droit routier français !"

        elif intent == Intent.LEGAL_QUERY:
            # Full RAG pipeline
            with st.spinner("Analyse de la jurisprudence..."):
                results = retriever.search(prompt, k=3)
                relevant_docs = [
                    r for r in results
                    if r.score < settings.RELEVANCE_THRESHOLD
                ]

        # intent == CHITCHAT → relevant_docs stays empty, generator handles naturally

        # --- Display Sources ---
        if relevant_docs:
            with st.expander(f"📚 {len(relevant_docs)} Articles de loi consultés", expanded=False):
                for r in relevant_docs:
                    st.markdown(f"**{r.article.article_number}**")
                    st.caption(f"{r.article.content[:250]}...")
                    st.markdown(f"[Lire sur Légifrance]({r.article.full_url})")
                    st.divider()

        # --- Generation (Streaming) ---
        if not full_response:
            try:
                for chunk in generator.generate_stream(prompt, relevant_docs):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            except Exception as e:
                st.error(f"Generation Error: {e}")
                full_response = "Une erreur est survenue lors de la génération."

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
