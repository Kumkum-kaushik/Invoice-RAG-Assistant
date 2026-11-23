
import os
import json
import math
import asyncio
import pandas as pd

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall
)

from .vectorstore import INDEXES, retrieve
from .ragas_evaluation_dataset import ragas_evaluation_data


TOP_K = 3
STRATEGIES = ("semantic", "hybrid")
RESULTS_DIR = "results"

# Groq model available to this account (llama-3.3-70b-versatile is not)
EVALUATOR_MODEL = "openai/gpt-oss-120b"


load_dotenv(".env")

# Optional judge guidance (Issue 3). Without it, the judge often marks a
# context "not useful" when it supports only ONE of several invoices in the
# reference (its stated reason: "the answer lists several other invoices").
# RAGAS context precision is meant to measure whether each chunk is relevant,
# so this instruction restates that definition. It is OFF by default; run
# with RAGAS_JUDGE_GUIDANCE=1 to get the guided variant, saved separately.
JUDGE_GUIDANCE = os.getenv("RAGAS_JUDGE_GUIDANCE") == "1"
GUIDANCE_PROMPT = (
    "When asked whether a context was useful in arriving at an answer: the "
    "context is useful (verdict 1) if it supports at least one statement of "
    "the answer. It does not need to support every statement."
)
VARIANT = "guided" if JUDGE_GUIDANCE else "default"
LOG_PATH = f"{RESULTS_DIR}/ragas_log_{VARIANT}.jsonl"

# 1. Connect to Groq (OpenAI-compatible endpoint)
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# gpt-oss is a reasoning model: its reasoning tokens count against
# max_tokens, and RAGAS's default of 1024 truncated the ContextRecall
# output for 6-statement references (IncompleteOutputException).
MAX_TOKENS = 4096

evaluator_llm = llm_factory(
    EVALUATOR_MODEL,
    client=client,
    max_tokens=MAX_TOKENS,
    **({"system_prompt": GUIDANCE_PROMPT} if JUDGE_GUIDANCE else {})
)


# 2. Record every judge output and retry on rate limits.
#    RAGAS 0.4.4 metrics call llm.agenerate(prompt, OutputModel) once per
#    context (ContextPrecision) or once per record (ContextRecall) and only
#    return the final number, so the verdicts/reasons are captured here.
judge_calls = []
_original_agenerate = evaluator_llm.agenerate


async def logged_agenerate(prompt, response_model, *args, **kwargs):

    budget = MAX_TOKENS

    for attempt in range(6):
        evaluator_llm.model_args["max_tokens"] = budget
        try:
            output = await _original_agenerate(
                prompt, response_model, *args, **kwargs)
            judge_calls.append(output.model_dump())
            return output
        except Exception as error:
            text = str(error).lower()

            if ("429" in text or "rate limit" in text
                    or "rate_limit" in text) and attempt < 5:
                wait = 15 * (attempt + 1)
                print(f"    rate limited, retrying in {wait}s")
                await asyncio.sleep(wait)
                continue

            # Truncated structured output: retry ONCE with double budget.
            # If it still fails the error propagates and the score is NaN.
            if (("max_tokens" in text or "incomplete" in text)
                    and budget == MAX_TOKENS):
                budget = MAX_TOKENS * 2
                print(f"    output truncated, retrying with "
                      f"max_tokens={budget}")
                continue

            raise
        finally:
            evaluator_llm.model_args["max_tokens"] = MAX_TOKENS


evaluator_llm.agenerate = logged_agenerate


# 3. RAGAS 0.4.4 metrics. Both need: user_input, retrieved_contexts and a
#    reference ANSWER (not reference contexts).
context_precision = ContextPrecision(llm=evaluator_llm)
context_recall = ContextRecall(llm=evaluator_llm)


# 4. Retrieve contexts
def collect_contexts(item, index_path, metadata_path, strategy):

    results = retrieve(
        item["question"], index_path, metadata_path,
        top_k=TOP_K, strategy=strategy, diversify=True,
    )

    contexts = [
        f"Source: {r['source']} (page {r['page_number']}, "
        f"section {r['section']})\nContent: {r['text']}"
        for r in results
    ]

    return results, contexts


async def score_metric(metric, item, contexts):
    # Returns (score, judge outputs, error). A failure is recorded as NaN,
    # never as 0, so it cannot be mistaken for a bad retrieval.
    start = len(judge_calls)
    try:
        result = await metric.ascore(
            user_input=item["question"],
            reference=item["reference"],
            retrieved_contexts=contexts,
        )
        return float(result.value), judge_calls[start:], None
    except Exception as error:
        return math.nan, judge_calls[start:], f"{type(error).__name__}: {error}"


# 5. Evaluate one question
async def evaluate_record(item, index_path, metadata_path, strategy, log):

    results, contexts = collect_contexts(
        item, index_path, metadata_path, strategy)

    expected = set(item["expected_invoices"])

    print("\nQUESTION :", item["question"])
    print("EXPECTED :", sorted(expected))
    for r in results:
        status = "RELEVANT" if r["source"] in expected else "not relevant"
        print(f"  {r['rank']}. {r['source']:<20} score={r['score']:.4f} "
              f"match={r['match']:<8} {status}")

    precision, precision_judge, precision_error = await score_metric(
        context_precision, item, contexts)
    recall, recall_judge, recall_error = await score_metric(
        context_recall, item, contexts)

    print(f"  Context Precision: {precision:.3f}"
          + (f"  ERROR {precision_error}" if precision_error else ""))
    print(f"  Context Recall   : {recall:.3f}"
          + (f"  ERROR {recall_error}" if recall_error else ""))

    # Judge's per-context verdicts, to see WHY a score is low
    for i, call in enumerate(precision_judge, start=1):
        print(f"    precision verdict ctx{i}: {call.get('verdict')} "
              f"- {str(call.get('reason'))[:140]}")
    for call in recall_judge:
        for c in call.get("classifications", []):
            print(f"    recall {c.get('attributed')}: "
                  f"{str(c.get('statement'))[:100]}")

    log.write(json.dumps({
        "strategy": strategy,
        "index": index_path,
        "question": item["question"],
        "type": item.get("type"),
        "reference": item["reference"],
        "expected_invoices": sorted(expected),
        "retrieved": [
            {k: r[k] for k in ("rank", "source", "page_number", "chunk_id",
                               "section", "score", "match")}
            for r in results
        ],
        "contexts": contexts,
        "context_precision": precision,
        "context_recall": recall,
        "precision_judge": precision_judge,
        "recall_judge": recall_judge,
        "errors": [e for e in (precision_error, recall_error) if e],
    }, ensure_ascii=False, default=str) + "\n")
    log.flush()

    return {
        "question": item["question"],
        "context_precision": precision,
        "context_recall": recall,
        "errors": int(bool(precision_error)) + int(bool(recall_error)),
    }


def mean_ignoring_nan(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


# 6. Main execution
async def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary_rows = []
    detail_rows = []

    with open(LOG_PATH, "w", encoding="utf-8") as log:

        for strategy in STRATEGIES:
            for chunking, (index_path, metadata_path) in INDEXES.items():

                print(f"\n========== {strategy.upper()} / "
                      f"{chunking.upper()} ==========")

                rows = []
                for item in selected_items():
                    rows.append(await evaluate_record(
                        item, index_path, metadata_path, strategy, log))

                for row in rows:
                    detail_rows.append({
                        "Retrieval Strategy": strategy.capitalize(),
                        "Chunking": chunking, **row})

                summary_rows.append({
                    "Retrieval Strategy": strategy.capitalize(),
                    "Chunking": chunking,
                    "Context Precision": mean_ignoring_nan(
                        [r["context_precision"] for r in rows]),
                    "Context Recall": mean_ignoring_nan(
                        [r["context_recall"] for r in rows]),
                    "Failed evaluations": sum(r["errors"] for r in rows),
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{RESULTS_DIR}/ragas_metrics_{VARIANT}.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(
        f"{RESULTS_DIR}/ragas_per_question_{VARIANT}.csv", index=False)

    print("\n========== FINAL RAGAS COMPARISON (top_k = 3) ==========")
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved {RESULTS_DIR}/ragas_metrics_{VARIANT}.csv, "
          f"{RESULTS_DIR}/ragas_per_question_{VARIANT}.csv and {LOG_PATH}")


def selected_items():
    # The full set is 50 questions x 4 strategy/chunking runs, which needs
    # far more than the Groq free tier's 200k tokens/day for gpt-oss-120b.
    # RAGAS_PER_CATEGORY=N takes the first N questions of each category
    # (stable IDs, so the subset is reproducible). Unset = all questions.
    per_category = os.getenv("RAGAS_PER_CATEGORY")
    if not per_category:
        return ragas_evaluation_data
    taken, out = {}, []
    for item in ragas_evaluation_data:
        category = item.get("category", "uncategorized")
        if taken.get(category, 0) < int(per_category):
            taken[category] = taken.get(category, 0) + 1
            out.append(item)
    return out


if __name__ == "__main__":
    print(f"RAGAS questions per run: {len(selected_items())} "
          f"(judge variant: {VARIANT})")
    asyncio.run(main())
