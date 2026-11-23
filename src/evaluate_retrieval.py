
import os
import pandas as pd

from .evaluation_dataset import evaluation_data
from .vectorstore import INDEXES, retrieve


K_VALUES = (1, 3, 5, 10)
MULTI_DOC_K = (3, 5, 10)
RESULTS_DIR = "results"


def retrieved_invoices(results):
    # Unique invoice names in rank order (several chunks may share one)
    invoices = []
    for result in results:
        if result["source"] not in invoices:
            invoices.append(result["source"])
    return invoices


def metrics_at_k(retrieved, expected, k):
    """
    Invoice-level metrics for one question.

    hit       = 1 if any relevant invoice is in the top k
    precision = relevant retrieved / retrieved (fewer than k may come back)
    recall    = relevant retrieved / all relevant
    rr        = 1 / rank of the first relevant invoice (0 if none in top k)
    """
    top = retrieved[:k]
    relevant = [invoice for invoice in top if invoice in expected]

    first_rank = next(
        (rank for rank, invoice in enumerate(top, start=1)
         if invoice in expected),
        None,
    )

    return {
        f"hit@{k}": int(bool(relevant)),
        f"precision@{k}": len(relevant) / len(top) if top else 0.0,
        f"recall@{k}": len(relevant) / len(expected) if expected else 0.0,
        f"mrr@{k}": 1.0 / first_rank if first_rank else 0.0,
    }


def evaluate_retriever(index_path, metadata_path, strategy="hybrid",
                       k_values=K_VALUES, verbose=True):

    max_k = max(k_values)
    rows = []

    for item in evaluation_data:

        query = item["question"]
        expected = set(item["expected_invoices"])

        # diversify=True: the top max_k chunks come from distinct invoices
        results = retrieve(
            query, index_path, metadata_path,
            top_k=max_k, strategy=strategy, diversify=True,
        )
        invoices = retrieved_invoices(results)

        row = {"id": item.get("id"), "category": item.get("category"),
               "question": query, "num_expected": len(expected),
               "retrieved": invoices}
        for k in k_values:
            row.update(metrics_at_k(invoices, expected, k))
        rows.append(row)

        if verbose:
            print("\nQuestion:", query)
            print(f"Expected ({len(expected)}):",
                  sorted(expected) if len(expected) <= 6
                  else f"{sorted(expected)[:6]} ...")
            for result in results:
                status = "RELEVANT" if result["source"] in expected \
                    else "not relevant"
                print(f"  {result['rank']}. {result['source']:<20} "
                      f"score={result['score']:.4f} "
                      f"match={result['match']:<8} {status}")
            print("  " + "  ".join(
                f"{name}={row[name]:.2f}"
                for name in ("hit@3", "precision@3", "recall@3", "mrr@3")))

    return rows


def summarize(rows, k_values=K_VALUES):
    metric_names = [f"{m}@{k}" for k in k_values
                    for m in ("hit", "precision", "recall", "mrr")]
    return {name: sum(r[name] for r in rows) / len(rows)
            for name in metric_names}


if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_rows = []
    detail_rows = []

    for strategy in ("semantic", "hybrid"):
        for chunking, (index_path, metadata_path) in INDEXES.items():

            print(f"\n========== {strategy.upper()} / {chunking.upper()} ==========")

            rows = evaluate_retriever(index_path, metadata_path, strategy)

            for row in rows:
                detail_rows.append({
                    "Retrieval Strategy": strategy.capitalize(),
                    "Chunking": chunking,
                    **{k: v for k, v in row.items() if k != "retrieved"},
                    "retrieved": ", ".join(row["retrieved"]),
                })

            summary_rows.append({
                "Retrieval Strategy": strategy.capitalize(),
                "Chunking": chunking,
                **summarize(rows),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{RESULTS_DIR}/retrieval_metrics.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(
        f"{RESULTS_DIR}/retrieval_per_question.csv", index=False)

    print("\n========== RETRIEVAL SUMMARY (invoice level) ==========")
    print(summary.round(3).to_string(index=False))

    # Per-category view: where each strategy wins or fails
    by_category = (
        pd.DataFrame(detail_rows)
        .groupby(["category", "Retrieval Strategy", "Chunking"])
        [["hit@3", "precision@3", "recall@3", "mrr@3", "recall@10"]]
        .mean().reset_index()
    )
    by_category.to_csv(f"{RESULTS_DIR}/retrieval_by_category.csv",
                       index=False)

    print("\n========== HIT@3 / MRR@3 BY QUESTION CATEGORY ==========")
    print(by_category.pivot_table(
        index="category", columns=["Retrieval Strategy", "Chunking"],
        values=["hit@3", "mrr@3"],
    ).round(2).to_string())

    # Multi-invoice questions: how Precision@K / Recall@K move with K
    details = pd.DataFrame(detail_rows)
    multi = details[(details["num_expected"] > 1)
                    & (details["num_expected"] <= 10)]
    multi_rows = []
    for _, row in multi.iterrows():
        for k in MULTI_DOC_K:
            multi_rows.append({
                "Retrieval Strategy": row["Retrieval Strategy"],
                "Chunking": row["Chunking"],
                "question": row["question"],
                "relevant": row["num_expected"],
                "K": k,
                "Precision@K": row[f"precision@{k}"],
                "Recall@K": row[f"recall@{k}"],
            })
    multi_table = pd.DataFrame(multi_rows)
    multi_table.to_csv(f"{RESULTS_DIR}/multi_doc_topk.csv", index=False)

    print("\n========== MULTI-INVOICE QUESTIONS: TOP-K 3 vs 5 vs 10 ==========")
    print(multi_table.pivot_table(
        index=["question", "Retrieval Strategy", "Chunking"],
        columns="K", values=["Precision@K", "Recall@K"],
    ).round(2).to_string())
    print(f"\nSaved {RESULTS_DIR}/retrieval_metrics.csv "
          f"and {RESULTS_DIR}/retrieval_per_question.csv")
