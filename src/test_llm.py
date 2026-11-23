from src.llm_client import generate_response

prompt = """
You are an invoice assistant.

Answer the question using only the context below.
If the answer is not available, say:
"I couldn't find this information in the provided invoices."

Context:
Order ID: 11062
Customer ID: REGGC
Contact Name: Maurizio Moroni
Country: Italy

Question:
Who placed Order ID 11062?
"""

answer = generate_response(prompt)

print("LLM Response:")
print(answer)