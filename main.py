"""
Traffic Law RAG — CLI Entry Point.

Orchestrates the interaction between user, Retrieval system,
and Generative AI via a terminal interface.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src.config import settings
    from src.retrieval import TrafficRetriever
    from src.generation import TrafficGenerator
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    print("Ensure you are running from the project root: 'poetry run python main.py'")
    sys.exit(1)

# --- Logging ---
logging.basicConfig(level=logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)


def setup_system() -> Optional[Tuple[TrafficRetriever, TrafficGenerator]]:
    """Initializes RAG components. Returns None on failure."""
    print("⚙️  Initializing System Components...")

    if not settings.CHROMA_DB_PATH.exists():
        print(f"❌ Error: Vector Database not found at {settings.CHROMA_DB_PATH}")
        print("👉 Action required: Run 'poetry run python src/ingestion/indexing.py'")
        return None

    try:
        retriever = TrafficRetriever()
        generator = TrafficGenerator()
        print("✅ System Ready. Knowledge Base loaded.")
        return retriever, generator

    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return None


def run_chat_loop(retriever: TrafficRetriever, generator: TrafficGenerator):
    """Main interactive loop."""
    print("\n" + "=" * 50)
    print("🚗  AI CODE DE LA ROUTE (Assistant Juridique)")
    print("    Type 'q', 'quit' or 'exit' to leave.")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("❓ Votre question : ").strip()

            if query.lower() in ['q', 'quit', 'exit']:
                print("👋 Au revoir. Drive safe!")
                break

            if len(query) < 3:
                print("⚠️  Question trop courte, veuillez préciser.")
                continue

            print("   🔍 Analyse de la jurisprudence...", end="\r")
            results = retriever.search(query, k=3)
            print(" " * 50, end="\r")

            if not results:
                print("   ⚠️  Aucun article pertinent trouvé dans la base.")
                continue

            source_list = [r.article.article_number for r in results]
            print(f"   📚 Sources : {', '.join(source_list)}")

            print("\n   🤖 Réponse : ", end="", flush=True)

            for chunk in generator.generate_stream(query, results):
                print(chunk, end="", flush=True)

            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Arrêt forcé par l'utilisateur.")
            break
        except Exception as e:
            print(f"\n❌ Une erreur inattendue est survenue : {e}\n")


def main():
    """Application Entry Point."""
    components = setup_system()

    if components:
        retriever, generator = components
        run_chat_loop(retriever, generator)


if __name__ == "__main__":
    main()
