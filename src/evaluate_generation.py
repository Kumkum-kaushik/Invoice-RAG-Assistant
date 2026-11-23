
import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

from src.rag_pipeline import answer_question


load_dotenv()


# 1. Configure evaluator LLM using Groq
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

evaluator_llm = llm_factory(
    "openai/gpt-oss-120b",
    client=client
)


# 2. Configure evaluator embeddings
evaluator_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)


# 3. Initialize RAGAS metrics
faithfulness_metric = Faithfulness(
    llm=evaluator_llm
)

answer_relevancy_metric = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)


async def evaluate_generation(query):

    # Step A: Generate an answer using your RAG pipeline
    rag_result = answer_question(query)

    answer = rag_result["answer"]

    # Extract retrieved context texts
    contexts = [
        item["text"]
        for item in rag_result["retrieved_context"]
    ]

    if not contexts:
        print("No retrieved context. Skipping evaluation.")
        return

    print("\n========== GENERATED ANSWER ==========")
    print(answer)

    # Step B: Calculate Faithfulness
    faithfulness_result = await faithfulness_metric.ascore(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts
    )

    # Step C: Calculate Answer Relevancy
    relevancy_result = await answer_relevancy_metric.ascore(
        user_input=query,
        response=answer
    )

    # Step D: Display scores
    print("\n========== GENERATION EVALUATION ==========")

    print(
        "Faithfulness:",
        round(faithfulness_result.value, 4)
    )

    print(
        "Answer Relevancy:",
        round(relevancy_result.value, 4)
    )


if __name__ == "__main__":

    query = "Who placed Order ID 11062?"

    asyncio.run(evaluate_generation(query))
