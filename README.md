# 🧾 Intelligent Invoice RAG Assistant

An AI-powered invoice question-answering system built using **Retrieval-Augmented Generation (RAG)**. It enables users to ask natural-language questions about invoices and receive context-grounded answers using hybrid retrieval and an LLM.

The system combines semantic search, keyword-based retrieval, exact entity matching, and GPT-OSS to retrieve relevant invoice information and generate answers.

## 🚀 Key Features

- **PDF Extraction:** Extract invoice content from PDF documents using PyMuPDF.
- **Intelligent Chunking:** Supports fixed-size and structure-aware chunking strategies.
- **Semantic Search:** Generates vector embeddings using Sentence Transformers.
- **Vector Database:** Uses FAISS for similarity search.
- **Hybrid Retrieval:** Combines FAISS semantic search, BM25 lexical search, and exact entity matching.
- **LLM-based Generation:** Uses GPT-OSS 120B through Groq to generate answers from retrieved invoice context.
- **Hallucination Control:** Uses context-grounded prompting to reduce unsupported answers.
- **RAG Evaluation:** Evaluates retrieval and generation quality using RAGAS.

## 🏗️ Architecture

```text
                 Invoice PDF Documents
                          |
                          v
                   PDF Extraction
                       PyMuPDF
                          |
                          v
                     Text Chunking
               /                     \
       Fixed-size               Structure-aware
               \                     /
                          |
                          v
                   Text Embeddings
                Sentence Transformers
                          |
                          v
                     FAISS Index
                          |
                          v
                    User Question
                          |
                          v
                  Hybrid Retrieval
              /          |          \
          FAISS         BM25      Exact Matching
              \          |          /
                          |
                          v
                 Retrieved Context
                          |
                          v
                   Prompt Template
                          |
                          v
                  GPT-OSS 120B
                   (Groq API)
                          |
                          v
                  Generated Answer
                          |
                          v
                   RAGAS Evaluation
```

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| PDF Processing | PyMuPDF |
| Text Chunking | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Lexical Retrieval | BM25 |
| LLM | GPT-OSS 120B |
| LLM Provider | Groq |
| Evaluation | RAGAS |
| Environment | Python Virtual Environment |

## 📂 Project Structure

```text
Invoice-RAG-Assistant/
│
├── data/
│   ├── processed/
│   ├── embeddings/
│   └── vectorstore/
│
├── src/
│   ├── extraction.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── llm_client.py
│   ├── rag_pipeline.py
│   ├── test_rag.py
│   ├── evaluate_retrieval.py
│   └── evaluate_generation.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: Adjust filenames to match the actual files in your repository.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Kumkum-kaushik/Invoice-RAG-Assistant.git

cd Invoice-RAG-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from [Groq Console](https://console.groq.com/).

Never commit your actual `.env` file or API key.

### 5. Prepare invoice data

Place your permitted invoice PDF files in the appropriate input directory and run the extraction, chunking, embedding, and indexing scripts.

The generated FAISS index and its metadata must be available before running the RAG pipeline.

## ▶️ Run the RAG Assistant

Run the question-answering pipeline:

```bash
python -m src.test_rag
```

Example question:

```text
Who placed Order ID 11062?
```

Example answer:

```text
The order was placed by Maurizio Moroni
(Customer ID REGGC).
```

## 📊 Evaluation Framework

The project evaluates both retrieval and generation quality.

### Retrieval Evaluation

| Metric | Purpose |
|---|---|
| Context Precision | Measures the relevance of retrieved context |
| Context Recall | Measures how much relevant information was retrieved |

### Generation Evaluation

| Metric | Purpose |
|---|---|
| Faithfulness | Measures whether generated claims are supported by retrieved context |
| Answer Relevancy | Measures whether the answer addresses the user's question |

Run generation evaluation:

```bash
python -m src.evaluate_generation
```

Retrieval evaluation is implemented separately.

## 🔍 Key Technical Challenges

- Exact invoice identifiers are difficult to retrieve through semantic similarity alone.
- Fixed-size chunking can separate invoice fields from their context.
- Structure-aware chunking preserves invoice sections and their relationships.
- Hybrid retrieval combines semantic understanding with exact keyword matching.
- LLM generation must remain grounded in retrieved invoice information.
- RAGAS evaluation helps identify retrieval and generation quality issues.

## 🔮 Future Improvements

- Interactive Streamlit-based chat interface.
- Conversation memory for follow-up invoice queries.
- Source citations with invoice filename and page number.
- Retrieval optimization and reranking.
- More comprehensive evaluation datasets.
- Improved hallucination detection and answer validation.

## 👩‍💻 Author

**Kumkum Kaushik**

AI/ML Engineer | Generative AI | RAG | AI Agents

- GitHub: https://github.com/Kumkum-kaushik
- LinkedIn: https://www.linkedin.com/in/kumkum-kaush
