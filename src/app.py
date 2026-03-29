import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.classifier import Intent

st.set_page_config(page_title="LégiRoute AI", page_icon="⚖️", layout="wide")
st.title("⚖️ LégiRoute")
st.caption("Assistant Juridique — Code de la Route Français")


@st.cache_resource
def load_rag():
    from src.rag import RAG
    if not settings.CHROMA_DB_PATH.exists():
        from src.ingestion.indexing import main as run_indexing
        run_indexing()
    return RAG()


with st.spinner("Chargement de la base juridique..."):
    rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Posez-moi une question sur le Code de la Route."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ex: Sanction pour téléphone au volant ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        intent = rag.classifier.classify(prompt)

        if intent == Intent.OFF_TOPIC:
            full_response = "Je suis spécialisé dans le Code de la Route. Je ne peux pas répondre à cette question."
        else:
            sources = []
            if intent == Intent.LEGAL_QUERY:
                with st.spinner("Recherche..."):
                    results = rag.retriever.search(prompt, k=3)
                    sources = [r for r in results if r.score < settings.RELEVANCE_THRESHOLD]

                if sources:
                    with st.expander(f"📚 {len(sources)} articles consultés"):
                        for r in sources:
                            st.markdown(f"[**{r.article.article_number}**]({r.article.full_url})")
                            st.caption(r.article.content[:250] + "...")
                            st.divider()

            for chunk in rag.generator.generate_stream(prompt, sources):
                full_response += chunk
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
