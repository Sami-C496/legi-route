import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings

logging.basicConfig(level=logging.ERROR)


def main():
    from src.rag import RAG
    rag = RAG()

    print("\n" + "=" * 50)
    print("⚖️  LégiRoute — Code de la Route")
    print("   Type 'q' to quit.")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("❓ ").strip()

            if query.lower() in ("q", "quit", "exit"):
                print("👋 Au revoir.")
                break

            if len(query) < 3:
                continue

            for chunk in rag.stream(query):
                print(chunk, end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\n👋")
            break


if __name__ == "__main__":
    main()
