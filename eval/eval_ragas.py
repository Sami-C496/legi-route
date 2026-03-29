"""
RAGAS Evaluation for LégiRoute.

Runs the full RAG pipeline on the evaluation dataset, then scores
each response with RAGAS metrics (Gemini as judge).

Usage:
    poetry run python eval/eval_ragas.py
    poetry run python eval/eval_ragas.py --k 5
    poetry run python eval/eval_ragas.py --no-cache
"""

import csv
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_questions(filepath: Path) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_ragas_llm():
    from src.config import settings
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GOOGLE_API_KEY,
    )
    return LangchainLLMWrapper(chat_model)


def run_rag_pipeline(questions: list[dict], k: int, cache_path: Path = None) -> list[SingleTurnSample]:
    if cache_path and cache_path.exists():
        print(f"📦 Loading cached RAG results from {cache_path.name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return [
            SingleTurnSample(
                user_input=s["user_input"],
                response=s["response"],
                retrieved_contexts=s["retrieved_contexts"],
            )
            for s in cached
        ]

    from src.rag import RAG
    rag = RAG()
    samples = []
    raw_cache = []

    for i, q in enumerate(questions):
        query = q["query"]
        print(f"  [{i+1}/{len(questions)}] {query[:60]}...")

        try:
            result = rag.query(query, k=k)
            contexts = result.contexts if result.contexts else ["Aucun contexte."]
            samples.append(SingleTurnSample(
                user_input=query, response=result.response, retrieved_contexts=contexts,
            ))
            raw_cache.append({"user_input": query, "response": result.response, "retrieved_contexts": contexts})
        except Exception as e:
            logger.warning(f"Failed on '{query}': {e}")
            samples.append(SingleTurnSample(
                user_input=query, response=f"Erreur: {e}", retrieved_contexts=["Erreur."],
            ))
            raw_cache.append({"user_input": query, "response": f"Erreur: {e}", "retrieved_contexts": ["Erreur."]})

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(raw_cache, f, ensure_ascii=False, indent=2)
        print(f"💾 RAG results cached to {cache_path.name}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for LégiRoute")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    eval_dir = Path(__file__).resolve().parent
    dataset_path = Path(args.dataset) if args.dataset else eval_dir / "test_questions.csv"
    output_path = eval_dir / "ragas_results.json"
    cache_path = eval_dir / "rag_cache.json"

    if args.no_cache and cache_path.exists():
        cache_path.unlink()

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return

    questions = load_questions(dataset_path)
    print(f"📋 Loaded {len(questions)} questions")

    # 1. RAG pipeline
    print(f"\n🔍 Running RAG (k={args.k})...")
    samples = run_rag_pipeline(questions, args.k, cache_path=cache_path)
    dataset = EvaluationDataset(samples=samples)

    # 2. RAGAS scoring
    print("\n⚖️  Scoring with RAGAS...")
    ragas_llm = get_ragas_llm()

    metrics = [
        Faithfulness(),
        LLMContextPrecisionWithoutReference(),
    ]

    results = evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm)

    # 3. Display
    df = results.to_pandas()
    skip = {"user_input", "response", "retrieved_contexts", "category"}
    metric_cols = [c for c in df.columns if c not in skip]

    print("\n" + "=" * 60)
    print("📊 RAGAS RESULTS")
    print("=" * 60)

    for col in metric_cols:
        print(f"   {col:<40} {df[col].mean():.4f}")

    categories = [q["category"] for q in questions[:len(df)]]
    df["category"] = categories

    print("\n   By category:")
    for cat, group in sorted(df.groupby("category")):
        parts = [f"{col[:12]}={group[col].mean():.2f}" for col in metric_cols]
        print(f"     {cat:<18} {' | '.join(parts)}  ({len(group)}q)")

    # 4. Save
    output = {
        "metadata": {
            "k": args.k,
            "total_questions": len(questions),
            "timestamp": datetime.now().isoformat(),
            "metrics": metric_cols,
        },
        "summary": {
            col: round(float(df[col].mean()), 4) for col in metric_cols
        },
        "details": json.loads(df.to_json(orient="records")),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Saved to {output_path}")


if __name__ == "__main__":
    main()
