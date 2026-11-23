
from src.vectorstore import retrieve, INDEXES
from src.llm_client import generate_response


def answer_question(
    query,
    index_name="Structure-aware",
    strategy="hybrid",
    top_k=5
):
    # 1. Get the index and metadata paths
    index_path, metadata_path = INDEXES[index_name]

    # 2. Retrieve relevant invoice chunks
    results = retrieve(
        query=query,
        index_path=index_path,
        metadata_path=metadata_path,
        top_k=top_k,
        strategy=strategy
    )

    # 3. Handle empty retrieval
    if not results:
        return {
            "question": query,
            "retrieved_context": [],
            "answer": "I couldn't find relevant information in the invoices."
        }

    # 4. Prepare context with invoice information
    context = "\n\n".join(
        f"Invoice: {result['source']}\n"
        f"Page: {result['page_number']}\n"
        f"Content: {result['text']}"
        for result in results
    )

    # 5. Create the prompt
    prompt = f"""
You are an intelligent Invoice Question Answering Assistant.

Your task is to answer questions about invoices.

Follow these rules:
1. Answer using only the provided context.
2. Do not invent or assume any information.
3. If the answer is unavailable, clearly say so.
4. Keep your answer clear and concise.
5. Mention the invoice filename when relevant.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    # 6. Generate answer using GPT-OSS
    answer = generate_response(prompt)

    return {
        "question": query,
        "retrieved_context": results,
        "answer": answer
    }
