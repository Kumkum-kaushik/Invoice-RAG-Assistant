import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall
)

load_dotenv()

# Connect to Groq
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Create evaluator LLM
evaluator_llm = llm_factory(
    "llama-3.3-70b-versatile",
    client=client
)

# Initialize evaluation metrics
context_precision = ContextPrecision(llm=evaluator_llm)
context_recall = ContextRecall(llm=evaluator_llm)


async def evaluate_record(item):

    precision = await context_precision.ascore(
        user_input=item["question"],
        reference=item["reference"],
        retrieved_contexts=item["retrieved_contexts"]
    )

    recall = await context_recall.ascore(
        user_input=item["question"],
        reference=item["reference"],
        retrieved_contexts=item["retrieved_contexts"]
    )

    return precision.value, recall.value