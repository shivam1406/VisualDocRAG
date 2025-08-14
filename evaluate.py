# evaluate.py
import json
import time
from src.pipeline import VisualDocRAG

TEST_QUERIES = [
    {
        "query": "What was the revenue in Q3?",
        "gold_answer": "17000"
    },
    {
        "query": "What is the total amount due?",
        "gold_answer": "$1200"
    },
    {
        "query": "Which quarter had the highest growth?",
        "gold_answer": "Q3"
    },
    {
        "query": "List all bullet points in the flyer.",
        "gold_answer": "- AI and Machine Learning Talks; - Cloud Computing Workshops; - Networking Opportunities"
    }
]

def run_evaluation():
    rag = VisualDocRAG()  # Loads existing vectorstore
    results = []

    for item in TEST_QUERIES:
        q = item["query"]
        gold = item.get("gold_answer", "")

        start_time = time.time()
        res = rag.query(q)
        latency = round((time.time() - start_time) * 1000, 2)

        results.append({
            "query": q,
            "predicted_answer": res["answer"],
            "contexts": [
                {
                    "text": ctx["text"],
                    "metadata": ctx["metadata"],
                    "score": ctx.get("score")
                }
                for ctx in res["contexts"]
            ],
            "gold_answer": gold,
            "latency_ms": latency
        })

        print(f"[{latency} ms] {q} -> {res['answer'][:100]}...")

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n✅ Evaluation complete. Results saved to `evaluation_results.json`.")

if __name__ == "__main__":
    run_evaluation()