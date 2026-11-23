
from src.rag_pipeline import answer_question

# Ask a question
query = "Who placed Order ID 11062?"

# Execute the RAG pipeline
result = answer_question(query)

# Display results
print("\n========== RAG RESULTS ==========")

print("\nQuestion:")
print(result["question"])

print("\nRetrieved Invoices:")
for item in result["retrieved_context"]:
    print("Invoice:", item["source"])
    print("Match:", item.get("match"))
    print("Content:", item["text"][:200])
    print("-------------------------")

print("\nGenerated Answer:")
print(result["answer"])
