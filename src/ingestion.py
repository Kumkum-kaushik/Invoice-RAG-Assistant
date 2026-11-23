
import fitz
import os
import json


def extract_text(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text("text")

        pages.append({
            "page_number": page_number,
            "text": text
        })

    doc.close()

    return pages


def process_invoices(folder_path):

    documents = []

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(folder_path, filename)

            pages = extract_text(pdf_path)

            for page in pages:

                documents.append({
                    "source": filename,
                    "page_number": page["page_number"],
                    "text": page["text"]
                })

    return documents


if __name__ == "__main__":

    documents = process_invoices("data/invoices")

    os.makedirs("data/processed", exist_ok=True)

    with open(
        "data/processed/extracted_invoices.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Total extracted pages:", len(documents))
    print("Total documents:", len(set(
        doc["source"] for doc in documents
    )))
    print("Extraction completed!")
