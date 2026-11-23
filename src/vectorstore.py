import os
import re
import json
import unicodedata
import faiss
import numpy as np

from dataclasses import dataclass, field
from functools import lru_cache
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# Load the same embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Index / metadata pairs, shared by the evaluation scripts
INDEXES = {
    "Fixed-size": (
        "data/vectorstore/fixed_index.index",
        "data/embeddings/fixed_embeddings_metadata.json"
    ),
    "Structure-aware": (
        "data/vectorstore/structure_index.index",
        "data/embeddings/structure_embeddings_metadata.json"
    ),
}

STRATEGIES = ("semantic", "lexical", "hybrid")


def build_faiss_index(embedding_path, metadata_path, index_name):

    # Load embeddings
    embeddings = np.load(embedding_path)

    # FAISS expects float32 vectors
    embeddings = embeddings.astype("float32")

    # Load corresponding chunk metadata
    with open(metadata_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Validate alignment
    if len(embeddings) != len(chunks):
        raise ValueError("Embedding and metadata counts do not match!")

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # Add vectors to index
    index.add(embeddings)

    # Create output directory
    os.makedirs("data/vectorstore", exist_ok=True)

    # Save index
    index_path = f"data/vectorstore/{index_name}.index"

    faiss.write_index(index, index_path)

    print(f"\n{index_name} created successfully!")
    print("Total vectors:", index.ntotal)
    print("Vector dimensions:", dimension)

    return index


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize(text):
    # Case- and accent-insensitive form: "Pâté chinois" -> "pate chinois"
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.casefold()).strip()


def tokenize(text):
    # Lowercase, accent-free word tokens; keeps IDs like "11062" intact
    return re.findall(r"\w+", normalize(text))


# Invoice layout (see data/processed/extracted_invoices.json)
ORDER_ID_RE = re.compile(r"Order ID:\s*(\d+)")
CUSTOMER_ID_RE = re.compile(r"Customer ID:\s*([A-Z]+)")
# Product rows: "<product id>\n<name>\n<quantity>\n<unit price>"
PRODUCT_ROW_RE = re.compile(
    r"(?:^|\n)\d{1,3}\n([^\n]*[^\W\d_][^\n]*)\n\d+\n\d+(?:\.\d+)?(?=\n|$)"
)


# ---------------------------------------------------------------------------
# Store: FAISS index + metadata + BM25 + entity catalog, loaded once
# ---------------------------------------------------------------------------

@dataclass
class Store:
    index: object
    chunks: list
    bm25: BM25Okapi
    # source -> chunk positions, in metadata order
    source_chunks: dict = field(default_factory=dict)
    # entity value -> set of sources that carry it
    order_ids: dict = field(default_factory=dict)
    customer_ids: dict = field(default_factory=dict)
    # normalized product name -> display name
    products: dict = field(default_factory=dict)
    # normalized chunk text, for exact phrase matching
    norm_texts: list = field(default_factory=list)


@lru_cache(maxsize=None)
def load_store(index_path, metadata_path):

    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal != len(chunks):
        raise ValueError(
            f"{index_path} has {index.ntotal} vectors but "
            f"{metadata_path} has {len(chunks)} chunks"
        )

    store = Store(
        index=index,
        chunks=chunks,
        bm25=BM25Okapi([tokenize(chunk["text"]) for chunk in chunks]),
    )

    # Entity catalog, built from the data itself (nothing hardcoded).
    # Fixed-size chunks only carry the header in an invoice's first chunk,
    # so IDs are mapped to the SOURCE, then to all of that source's chunks.
    for position, chunk in enumerate(chunks):

        text = chunk["text"]
        source = chunk["source"]

        store.source_chunks.setdefault(source, []).append(position)
        store.norm_texts.append(normalize(text))

        for order_id in ORDER_ID_RE.findall(text):
            store.order_ids.setdefault(order_id, set()).add(source)

        for customer_id in CUSTOMER_ID_RE.findall(text):
            store.customer_ids.setdefault(customer_id, set()).add(source)

        for name in PRODUCT_ROW_RE.findall(text):
            store.products.setdefault(normalize(name), name.strip())

    return store


# ---------------------------------------------------------------------------
# Query understanding: which known entities does the query name?
# ---------------------------------------------------------------------------

def find_entities(query, store):

    entities = {"order_id": [], "customer_id": [], "product": []}

    # Order / invoice IDs: any number in the query that is a known order ID
    # ("Order ID 11062", "invoice 11062", "invoice_11062.pdf")
    for number in re.findall(r"\d+", query):
        if number in store.order_ids and number not in entities["order_id"]:
            entities["order_id"].append(number)

    # Customer IDs: an all-caps word, as written, that is a known customer ID.
    # Requiring caps avoids matching ordinary words such as "frank".
    for word in re.findall(r"\b[A-Z]{3,}\b", query):
        if word in store.customer_ids and word not in entities["customer_id"]:
            entities["customer_id"].append(word)

    # Product names: whole-phrase, case- and accent-insensitive match
    norm_query = f" {' '.join(tokenize(query))} "
    for norm_name in store.products:
        if f" {' '.join(tokenize(norm_name))} " in norm_query:
            entities["product"].append(norm_name)

    # A product name nested in a longer matched name is dropped
    entities["product"] = [
        name for name in entities["product"]
        if not any(name != other and name in other
                   for other in entities["product"])
    ]

    return entities


def exact_match_chunks(entities, store):
    # Chunks that contain what the query names exactly.
    # Several entity kinds narrow each other (intersection of invoices).

    source_sets = []

    if entities["order_id"]:
        source_sets.append(set().union(
            *(store.order_ids[i] for i in entities["order_id"])))

    if entities["customer_id"]:
        source_sets.append(set().union(
            *(store.customer_ids[i] for i in entities["customer_id"])))

    product_chunks = set()
    if entities["product"]:
        patterns = [
            re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)")
            for name in entities["product"]
        ]
        for position, norm_text in enumerate(store.norm_texts):
            if any(pattern.search(norm_text) for pattern in patterns):
                product_chunks.add(position)
        source_sets.append({store.chunks[p]["source"] for p in product_chunks})

    if not source_sets:
        return set()

    sources = set.intersection(*source_sets) or set.union(*source_sets)

    matched = set()
    for source in sources:
        for position in store.source_chunks[source]:
            # For products, only the chunks that list the product itself
            # (an ID query keeps every chunk of the invoice)
            if entities["product"] and not (
                    entities["order_id"] or entities["customer_id"]):
                if position not in product_chunks:
                    continue
            matched.add(position)

    return matched


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def encode_query(query):
    embedding = model.encode([query], normalize_embeddings=True)
    return np.array(embedding, dtype="float32")


def dense_ranking(query, index, k):

    # Search nearest vectors
    scores, indices = index.search(encode_query(query), min(k, index.ntotal))

    return [
        (int(idx), float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx != -1
    ]


def lexical_ranking(query, store, k):

    scores = store.bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:k]

    return [(int(idx), float(scores[idx])) for idx in order if scores[idx] > 0]


def to_result(store, position, score, **extra):

    chunk = store.chunks[position]

    return {
        "score": float(score),
        "source": chunk["source"],
        "page_number": chunk.get("page_number"),
        "chunk_id": chunk.get("chunk_id"),
        "section": chunk.get("section", "unknown"),
        "text": chunk["text"],
        "position": position,
        **extra,
    }


def dedupe_and_diversify(ranked, store, top_k, diversify):
    # ranked: list of positions, best first.
    # Drops repeated chunks (same source + same text); with diversify,
    # takes the best chunk of each invoice before any second chunk.

    seen_text = set()
    unique = []

    for position in ranked:
        chunk = store.chunks[position]
        key = (chunk["source"], chunk["text"])
        if key not in seen_text:
            seen_text.add(key)
            unique.append(position)

    if not diversify:
        return unique[:top_k]

    first, rest, seen_source = [], [], set()

    for position in unique:
        source = store.chunks[position]["source"]
        (rest if source in seen_source else first).append(position)
        seen_source.add(source)

    return (first + rest)[:top_k]


# ---------------------------------------------------------------------------
# Public search functions
# ---------------------------------------------------------------------------

def search_faiss(query, index_path, metadata_path, top_k=3):

    # Pure semantic (embedding) search, unchanged behaviour
    store = load_store(index_path, metadata_path)

    return [
        to_result(store, idx, score, semantic_score=score, match="semantic")
        for idx, score in dense_ranking(query, store.index, top_k)
    ]


def retrieve(query, index_path, metadata_path, top_k=3, strategy="hybrid",
             candidates=50, rrf_k=60, exact_priority=True,
             restrict_to_exact=True, diversify=True):
    """
    Configurable retrieval.

    strategy:
        "semantic" - FAISS embeddings only
        "lexical"  - exact entity match, then BM25
        "hybrid"   - exact entity match, then RRF of FAISS + BM25

    exact_priority:    chunks of invoices named exactly in the query
                       (order ID, customer ID, product name) rank first
    restrict_to_exact: when exact matches exist, return only those
                       (fewer than top_k rather than unrelated filler)
    diversify:         one chunk per invoice before second chunks
    """

    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")

    store = load_store(index_path, metadata_path)
    n = store.index.ntotal

    entities = find_entities(query, store)
    exact = (
        exact_match_chunks(entities, store)
        if exact_priority and strategy != "semantic" else set()
    )
    matched_entities = [v for values in entities.values() for v in values]

    # Exact matches are re-ranked among themselves, so rank every chunk
    # when there are any; otherwise the top candidates are enough.
    depth = n if exact else max(candidates, top_k)

    semantic = dense_ranking(query, store.index, depth) \
        if strategy in ("semantic", "hybrid") else []
    lexical = lexical_ranking(query, store, depth) \
        if strategy in ("lexical", "hybrid") else []

    semantic_score = dict(semantic)
    lexical_score = dict(lexical)

    # Reciprocal Rank Fusion: ranks, not raw scores, are comparable
    fused = {}
    for ranking in (semantic, lexical):
        for rank, (position, _) in enumerate(ranking):
            fused[position] = fused.get(position, 0.0) + 1.0 / (rrf_k + rank + 1)

    if exact:
        pool = exact if restrict_to_exact else set(fused) | exact
        # exact matches first (tier), then fused score
        ranked = sorted(
            pool,
            key=lambda p: (p in exact, fused.get(p, 0.0)),
            reverse=True,
        )
    else:
        ranked = sorted(fused, key=fused.get, reverse=True)

    chosen = dedupe_and_diversify(ranked, store, top_k, diversify)

    results = []
    for rank, position in enumerate(chosen, start=1):

        score = (
            semantic_score.get(position, 0.0) if strategy == "semantic"
            else fused.get(position, 0.0)
        )

        results.append(to_result(
            store, position, score,
            rank=rank,
            strategy=strategy,
            match="exact" if position in exact else strategy,
            matched_entities=matched_entities if position in exact else [],
            semantic_score=semantic_score.get(position),
            lexical_score=lexical_score.get(position),
        ))

    return results


def hybrid_search(query, index_path, metadata_path, top_k=3,
                  candidates=50, rrf_k=60):

    # Kept for existing callers: hybrid strategy with default options
    return retrieve(
        query, index_path, metadata_path, top_k=top_k,
        strategy="hybrid", candidates=candidates, rrf_k=rrf_k,
    )


if __name__ == "__main__":

    # Build fixed-size index
    build_faiss_index(
        "data/embeddings/fixed_embeddings.npy",
        "data/embeddings/fixed_embeddings_metadata.json",
        "fixed_index"
    )

    # Build structure-aware index
    build_faiss_index(
        "data/embeddings/structure_embeddings.npy",
        "data/embeddings/structure_embeddings_metadata.json",
        "structure_index"
    )

    # Test semantic vs hybrid search
    query = "What was the total price of order 11045?"

    for name, (index_path, metadata_path) in INDEXES.items():
        for strategy in ("semantic", "hybrid"):

            print(f"\n{name.upper()} / {strategy.upper()} SEARCH")

            for result in retrieve(query, index_path, metadata_path,
                                   strategy=strategy):
                print("\nScore:", round(result["score"], 4),
                      "| Match:", result["match"])
                print("Source:", result["source"])
                print("Text:", result["text"])
