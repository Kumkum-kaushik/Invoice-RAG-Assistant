import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load extracted invoice documents
with open(
    "data/processed/extracted_invoices.json",
    "r",
    encoding="utf-8"
) as file:
    documents = json.load(file)


# Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)


all_chunks = []

for document in documents:

    text = document["text"]

    chunks = splitter.split_text(text)

    for index, chunk in enumerate(chunks):

        all_chunks.append({
            "source": document["source"],
            "page_number": document["page_number"],
            "chunk_id": index,
            "text": chunk
        })


# Save chunks
with open(
    "data/processed/fixed_chunks.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_chunks,
        file,
        indent=4,
        ensure_ascii=False
    )


print("Total original pages:", len(documents))
print("Total chunks:", len(all_chunks))

print("\nFirst chunk:")
print(all_chunks[0]["text"])
