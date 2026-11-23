
import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer


# Load embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(input_path, output_name):

    # Load chunks
    with open(input_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks found!")

    # Extract text
    texts = [chunk["text"] for chunk in chunks]

    # Convert text into vectors
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Create output directory
    os.makedirs("data/embeddings", exist_ok=True)

    # Save numerical vectors
    np.save(
        f"data/embeddings/{output_name}.npy",
        embeddings
    )

    # Preserve metadata
    with open(
        f"data/embeddings/{output_name}_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nEmbedding generation completed!")
    print("Total chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("Vector dimensions:", embeddings.shape[1])


if __name__ == "__main__":

    # Fixed-size chunk embeddings
    generate_embeddings(
        "data/processed/fixed_chunks.json",
        "fixed_embeddings"
    )

    # Structure-aware chunk embeddings
    generate_embeddings(
        "data/processed/structure_chunks.json",
        "structure_embeddings"
    )
