import json
from evaluation_dataset import evaluation_data
from vectorstore import search_faiss


def collect_contexts(index_path, metadata_path):

    evaluation_records = []

    for item in evaluation_data:

        query = item["question"]

        results = search_faiss(
            query,
            index_path,
            metadata_path,
            top_k=3
        )

        retrieved_contexts = [
            result["text"] for result in results
        ]

        evaluation_records.append({
            "question": query,
            "reference": item["reference"],
            "retrieved_contexts": retrieved_contexts
        })

    return evaluation_records


if __name__ == "__main__":

    fixed_records = collect_contexts(
        "data/vectorstore/fixed_index.index",
        "data/embeddings/fixed_embeddings_metadata.json"
    )

    structure_records = collect_contexts(
        "data/vectorstore/structure_index.index",
        "data/embeddings/structure_embeddings_metadata.json"
    )

    with open("fixed_contexts.json", "w") as f:
        json.dump(fixed_records, f, indent=4)

    with open("structure_contexts.json", "w") as f:
        json.dump(structure_records, f, indent=4)