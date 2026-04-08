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

_UNAVAILABLE_MSG = "Le service est momentanément surchargé. Veuillez réessayer dans quelques instants."

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
        sources = []

        try:
            intent = rag.classifier.classify(prompt)
        except Exception as e:
            logger.error("Classification unavailable: %s", e)
            intent = Intent.LEGAL_QUERY

        logger.info("query | intent=%s | turn=%d | q=%s", intent.value, len(history) // 2 + 1, prompt[:120])

        if intent == Intent.OFF_TOPIC:
            full_response = "Je suis spécialisé dans le Code de la Route. Je ne peux pas répondre à cette question."
        else:
            if intent == Intent.LEGAL_QUERY:
                with st.spinner("Recherche..."):
                    try:
                        search_query = rag.rewrite_query(prompt, history)
                        if search_query != prompt:
                            logger.info("rewrite | %s -> %s", prompt[:80], search_query[:80])
                        results = rag.retriever.search(search_query, k=3)
                        sources = [r for r in results if r.score > settings.RELEVANCE_THRESHOLD]
                        logger.info("retrieval | sources=%d | top_score=%.3f", len(sources), results[0].score if results else 0)
                    except Exception as e:
                        logger.error("Retrieval unavailable: %s", e)
                        full_response = _UNAVAILABLE_MSG

                if sources:
                    with st.expander(f"📚 {len(sources)} articles consultés"):
                        for r in sources:
                            st.markdown(f"[**{r.article.article_number}**]({r.article.full_url})")
                            text = r.article.content
                            st.caption(text[:250] + "..." if len(text) > 250 else text)
                            st.divider()

            if not full_response:
                try:
                    for chunk in rag.generator.generate_stream(prompt, sources, history=history):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                except Exception as e:
                    full_response = _UNAVAILABLE_MSG
                    logger.error("generation error | %s | q=%s", e, prompt[:80])

        placeholder.markdown(full_response)
        logger.info("done | intent=%s | sources=%d | time=%.2fs", intent.value, len(sources), time.monotonic() - t0)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
