
# Combines the outputs of the two separate evaluations:
#   python -m src.evaluate_retrieval                        -> results/retrieval_metrics.csv
#   python -m src.evaluate_ragas                            -> results/ragas_metrics_default.csv
#   RAGAS_JUDGE_GUIDANCE=1 python -m src.evaluate_ragas     -> results/ragas_metrics_guided.csv
# into results/evaluation_report.csv. Nothing is recomputed here.

import os
import pandas as pd

RESULTS_DIR = "results"
KEYS = ["Retrieval Strategy", "Chunking"]


if __name__ == "__main__":

    report = pd.read_csv(f"{RESULTS_DIR}/retrieval_metrics.csv")[
        KEYS + ["hit@3", "precision@3", "recall@3", "mrr@3"]
    ].rename(columns={"hit@3": "Hit@3", "precision@3": "Precision@3",
                      "recall@3": "Recall@3", "mrr@3": "MRR@3"})

    for variant in ("default", "guided"):
        path = f"{RESULTS_DIR}/ragas_metrics_{variant}.csv"
        if not os.path.exists(path):
            continue
        ragas = pd.read_csv(path).rename(columns={
            "Context Precision": f"Context Precision ({variant})",
            "Context Recall": f"Context Recall ({variant})",
            "Failed evaluations": f"Failed evals ({variant})",
        })
        report = report.merge(ragas, on=KEYS, how="left")

    report.to_csv(f"{RESULTS_DIR}/evaluation_report.csv", index=False)

    with pd.option_context("display.width", 200):
        print(report.round(3).to_string(index=False))
    print(f"\nSaved {RESULTS_DIR}/evaluation_report.csv")
