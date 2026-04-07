import logging
import time
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.classifier import Intent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="LégiRoute AI", page_icon="⚖️", layout="wide")
st.title("⚖️ LégiRoute")
st.caption("Assistant Juridique — Code de la Route Français")


@st.cache_resource
def load_rag():
    from src.rag import RAG
    return RAG()


with st.spinner("Chargement..."):
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

    history = st.session_state.messages[1:-1]
    t0 = time.monotonic()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        intent = rag.classifier.classify(prompt)
        logger.info("query | intent=%s | turn=%d | q=%s", intent.value, len(history) // 2 + 1, prompt[:120])

        if intent == Intent.OFF_TOPIC:
            full_response = "Je suis spécialisé dans le Code de la Route. Je ne peux pas répondre à cette question."
        else:
            sources = []
            if intent == Intent.LEGAL_QUERY:
                with st.spinner("Recherche..."):
                    search_query = rag.rewrite_query(prompt, history)
                    if search_query != prompt:
                        logger.info("rewrite | %s -> %s", prompt[:80], search_query[:80])
                    results = rag.retriever.search(search_query, k=3)
                    sources = [r for r in results if r.score > settings.RELEVANCE_THRESHOLD]
                    logger.info("retrieval | sources=%d | top_score=%.3f", len(sources), results[0].score if results else 0)

                if sources:
                    with st.expander(f"📚 {len(sources)} articles consultés"):
                        for r in sources:
                            st.markdown(f"[**{r.article.article_number}**]({r.article.full_url})")
                            text = r.article.content
                            st.caption(text[:250] + "..." if len(text) > 250 else text)
                            st.divider()

            for chunk in rag.generator.generate_stream(prompt, sources, history=history):
                full_response += chunk
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
        logger.info("done | intent=%s | sources=%d | time=%.2fs", intent.value, len(sources) if intent != Intent.OFF_TOPIC else 0, time.monotonic() - t0)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
