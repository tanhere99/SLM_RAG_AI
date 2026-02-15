# 🍏 SLM RAG AI (Local Mac Implementation)

A fully local **Retrieval Augmented Generation (RAG)** system designed to run efficiently on **Apple Silicon (M1/M2/M3)** using Small Language Models (SLM).

This project allows you to chat with your PDF documents using the **Phi-4-Mini** model — without sending any data to the cloud. Optimized for devices with **8GB RAM**.

---

## 🚀 Features

- ✅ **100% Local & Private**  
  No API keys (OpenAI / Anthropic) required.  
  Your data never leaves your laptop.

- ⚡ **Apple Silicon Optimized**  
  Uses **MLX** for high-performance inference on Mac.

- 🪶 **Lightweight**  
  Runs comfortably on **8GB RAM** using **4-bit quantization**.

- 🧰 **Tech Stack**
  - LangChain  
  - ChromaDB  
  - HuggingFace Embeddings  
  - MLX-LM  

---

## 🛠️ Prerequisites

### Hardware
- Apple Mac with **M1 / M2 / M3 chip**
- Minimum **8GB RAM**

### Software
- Python **3.10 or higher**

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/tanhere99/SLM_RAG_AI.git
cd SLM_RAG_AI
```

### 2️⃣ Create a Fresh Virtual Environment

> ⚠️ Highly recommended to avoid LangChain version conflicts.

```bash
python3 -m venv venv_rag
source venv_rag/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📖 Usage

### 1️⃣ Add Your Data

- Place your PDF file into the project root directory  
- Rename it to:

```
Story.pdf
```

OR update the path inside:

```
ingest_chroma.py
```

---

### 2️⃣ Create the Database

Run the ingestion script to split the PDF and store embeddings in the local vector store (ChromaDB).

```bash
python ingest_chroma.py
```

You should see:

```
✅ Ingestion Complete! Database saved to ./chroma_db
```

---

### 3️⃣ Chat with Your PDF

Launch the chat interface:

```bash
python chat_langchain.py
```

> 📥 First run will download the ~2.3GB Phi-4 model from HuggingFace.

---

## 💬 Example Interaction

```
You: Who is the main character?
AI: The main character is...
```

---

## 🔧 Troubleshooting

### ❌ "Model not found" or Import Errors
Make sure your virtual environment is activated:

```bash
source venv_rag/bin/activate
```

### ⚠️ Telemetry / Warning Messages

You may see:
- `Failed to send telemetry`
- `UNEXPECTED position_ids`

These are harmless logs from **ChromaDB** and **HuggingFace** and can be safely ignored.

### 🤖 Hallucinations or Repetition

If the model repeats responses:
- Ensure you're using the latest version of `chat_langchain.py`
- Confirm `<|user|>` stop tokens are implemented

---

## 🤝 Contributing

Feel free to:
- Open issues
- Submit pull requests
- Share improved prompts
- Suggest better MLX optimizations

---

## 📌 Notes

- Designed for **fully offline AI workflows**
- Ideal for privacy-focused document Q&A
- Optimized specifically for **Apple Silicon**
