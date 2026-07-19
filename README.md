# DSA RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot for learning Data Structures and Algorithms from a PDF revision guide.

The application retrieves relevant PDF passages, reranks them for relevance, and uses a local Ollama Mistral model to produce grounded answers. A Streamlit interface provides a simple chat experience with source pages, reranking scores, and an optional LLM-as-a-Judge evaluation.

## Features

- Loads and chunks `DS_Complete.pdf` using LangChain's `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- Generates local embeddings with Ollama's `all-minilm` model.
- Stores and searches PDF chunks locally in ChromaDB.
- Retrieves 10 semantic-search candidates per question.
- Uses FlashRank to rerank candidates and sends the best 4 chunks to Mistral.
- Generates context-grounded DSA answers with local Ollama Mistral.
- Displays PDF page citations and reranking scores.
- Includes basic guardrails for long inputs, prompt-injection attempts, and self-harm-related language.
- Shows an LLM-as-a-Judge evaluation for faithfulness, answer relevance, context relevance, and a pass/fail verdict.
- Provides a Streamlit chat UI with history, a clear-chat button, and expandable answer details.

## Architecture

```text
DS_Complete.pdf
  -> PDF loading and chunking
  -> all-minilm embeddings
  -> ChromaDB
  -> retrieve 10 candidates
  -> FlashRank reranks top 4
  -> Mistral generates a grounded answer
  -> sources, scores, and LLM-as-a-Judge evaluation
```

## Tech Stack

- Python 3.12+
- Ollama: `mistral` and `all-minilm`
- LangChain
- ChromaDB
- FlashRank
- Streamlit
- PyPDF
- uv

## Project Files

| File | Purpose |
| --- | --- |
| `Ragv2_DSA.py` | Ingests the PDF, creates chunks, and rebuilds the local ChromaDB database. |
| `Ragv2_Chatbot.py` | Terminal chatbot and reusable retrieval, reranking, guardrail, and judge functions. |
| `Frontend.py` | Streamlit web interface. |
| `DS_Complete.pdf` | Source DSA revision guide. |
| `chroma_db/` | Generated local vector database. |

## Setup

1. Install [Ollama](https://ollama.com/) and download the required models:

   ```powershell
   ollama pull mistral
   ollama pull all-minilm
   ```

2. Install the Python dependencies:

   ```powershell
   uv sync
   ```

## Run the Application

First, build or rebuild the local vector database. This removes the existing `chroma_db` folder and re-ingests the PDF.

```powershell
uv run python Ragv2_DSA.py
```

Run the terminal chatbot:

```powershell
uv run python Ragv2_Chatbot.py
```

Run the Streamlit frontend:

```powershell
uv run streamlit run Frontend.py
```

Then open the local URL shown by Streamlit, normally `http://localhost:8501`.

## Notes

- All models and the vector database run locally; no external LLM API key or Pinecone account is required.
- The LLM-as-a-Judge result is a development aid, not a substitute for human review.
- RAGAS evaluation is planned as a future offline benchmarking step.
