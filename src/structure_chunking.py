
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load extracted invoice documents
with open(
    "data/processed/extracted_invoices.json",
    "r",
    encoding="utf-8"
) as file:
    documents = json.load(file)


# Splitter for unusually large sections
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=80,
    separators=["\n\n", "\n", " ", ""]
)


def structure_aware_chunking(document):

    text = document["text"]

    # Identify invoice sections
    pattern = r"(?im)^\s*(Customer Details|Product Details):\s*$"

    matches = list(re.finditer(pattern, text))

    if not matches:
        # e.g. a continuation page with no section headers;
        # still keep source/page_number so search results can cite it
        return [{
            "source": document["source"],
            "page_number": document["page_number"],
            "section": "complete_invoice",
            "chunk_id": 0,
            "text": text
        }]

    # Extract invoice header
    header = text[:matches[0].start()].strip()

    sections = []

    # Add invoice information
    sections.append({
        "section": "invoice_information",
        "text": header
    })

    # Extract remaining sections
    for i, match in enumerate(matches):

        section_name = match.group(1).lower().replace(
            " ", "_"
        )

        start = match.end()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        section_text = text[start:end].strip()

        sections.append({
            "section": section_name,
            "text": section_text
        })

    # Preserve invoice identity
    chunks = []

    for section in sections:

        content = (
            f"Invoice Context:\n{header}\n\n"
            f"Section: {section['section']}\n"
            f"{section['text']}"
        )

        # Keep short sections together
        if len(content) <= 700:
            section_chunks = [content]
        else:
            section_chunks = splitter.split_text(content)

        for index, chunk in enumerate(section_chunks):

            chunks.append({
                "source": document["source"],
                "page_number": document["page_number"],
                "section": section["section"],
                "chunk_id": index,
                "text": chunk
            })

    return chunks


# Process all invoices
all_chunks = []

for document in documents:

    chunks = structure_aware_chunking(document)

    all_chunks.extend(chunks)


# Save results
with open(
    "data/processed/structure_chunks.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_chunks,
        file,
        indent=4,
        ensure_ascii=False
    )


print("Total original documents:", len(documents))
print("Total structure-aware chunks:", len(all_chunks))

print("\nFirst structure-aware chunk:")

print(all_chunks[0]["text"])
