# SLM RAG AI

**Small Language Model (SLM) + RAG** — Run a local RAG pipeline using a small, efficient LLM so you get document-grounded answers without cloud APIs or heavy hardware.

- **SLM**: [Phi-4 mini](https://huggingface.co/mlx-community/Phi-4-mini-instruct-4bit) (4-bit quantized) via **MLX** — fast on Apple Silicon, runs on CPU elsewhere.
- **RAG**: **LangChain** + **Chroma** — ingest PDFs, embed with HuggingFace, retrieve relevant chunks, and let the SLM answer from context.

Best of both: small footprint and private, local inference plus accurate, cited answers from your own documents.

## Features

- **Local SLM** — Phi-4 mini (4-bit) runs on your machine; no API keys or internet required for inference.
- **RAG pipeline** — PDF → chunks → Chroma embeddings → retrieval → SLM generates answers from context only.

## Setup

1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/tanhere99/SLM_RAG_AI.git
   cd SLM_RAG_AI
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add your PDF**
   - Place your PDF (e.g. `Story.pdf`) in the project root or adjust `PDF_PATH` in `venv_langchain/ingest_chroma.py`.

4. **Ingest documents into Chroma**
   ```bash
   cd venv_langchain
   python ingest_chroma.py
   ```

5. **Run the chat**
   ```bash
   python chat_langchain.py
   ```
   Type your questions; type `exit` or `quit` to stop.

## Project layout

- `venv_langchain/ingest_chroma.py` — **RAG ingestion**: load PDF, split text, embed and store in Chroma.
- `venv_langchain/chat_langchain.py` — **SLM + RAG**: query Chroma, then generate answers with the local Phi-4 SLM.

## Requirements

- Python 3.10+
- macOS with Apple Silicon recommended for MLX; CPU fallback on other platforms.
