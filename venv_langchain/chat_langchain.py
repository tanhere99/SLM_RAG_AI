import os
from typing import Any, List, Optional, Mapping

# --- IMPORTS ---
from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# MLX Imports
from mlx_lm import load, generate

# --- CONFIGURATION ---
DB_PATH = "./chroma_db" 
MODEL_ID = "mlx-community/Phi-4-mini-instruct-4bit"

# --- PHI-4 INSTRUCT PROMPT (CRITICAL FIX) ---
# We use the specific <|user|> and <|assistant|> tokens so the model knows 
# it is chatting, not writing a story.
prompt_template = """<|user|>
Use the following context to answer the question. 
Keep the answer concise (max 3 sentences). 
If you don't know, just say "I don't know".

Context: 
{context}

Question: 
{question}
<|end|>
<|assistant|>"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

class MLXPhi4(LLM):
    model_id: str = MODEL_ID
    model: Any = None
    tokenizer: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Loading {self.model_id}...")
        self.model, self.tokenizer = load(self.model_id)
        print("Model loaded.")

    @property
    def _llm_type(self) -> str:
        return "mlx_phi4"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Generate response
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=200, # Limit output length
            verbose=False,
        )

        # --- FORCEFUL CLEANING ---
        # 1. Cut off at standard stop tokens
        clean_response = response
        stop_signals = ["<|end|>", "<|endoftext|>", "<|user|>", "User:", "Context:"]
        
        for signal in stop_signals:
            if signal in clean_response:
                clean_response = clean_response.split(signal)[0]

        # 2. Remove any trailing newlines or whitespace
        clean_response = clean_response.strip()

        # 3. Fallback: If it still rambles with headers like "# Title", cut it there
        if "\n#" in clean_response:
             clean_response = clean_response.split("\n#")[0]

        return clean_response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_id": self.model_id}

def main():
    llm = MLXPhi4()
    
    # Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load Database
    print("Loading ChromaDB...")
    try:
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Create Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    print("\n--- RAG AI Ready (Type 'exit' to quit) ---")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]: break
        
        try:
            res = qa_chain.invoke({"query": query})
            print(f"\nAI: {res['result']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()