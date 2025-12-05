import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# -------------------------------------------------------------
# 1. SETUP AND MODEL LOADING
# -------------------------------------------------------------
REPO_ID = "abertekth/model"
FILENAME = "mio-modello-q4_k_m.gguf"

print(f"--- 1. Downloading LLM: {FILENAME} ---")
try:
    model_path = hf_hub_download(
        repo_id=REPO_ID, 
        filename=FILENAME
    )
except Exception as e:
    print(f"Error downloading model: {e}")
    raise e

print("--- Loading Llama into RAM ---")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,         # Context window
    n_threads=2,        # CPU threads
    n_gpu_layers=0,     # CPU only
    verbose=False
)

# -------------------------------------------------------------
# 2. RAG SETUP (HOPSWORKS KNOWLEDGE)
# -------------------------------------------------------------
print("--- 2. Preparing Hopsworks Knowledge Base ---")

TXT_FILE = "knowledge.txt"
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

knowledge_base = []
knowledge_embeddings = None

def load_knowledge_base():
    global knowledge_base, knowledge_embeddings
    
    if not os.path.exists(TXT_FILE):
        print(f"Warning: {TXT_FILE} not found. Creating dummy Hopsworks data.")
        text_data = "Hopsworks is a Feature Store for Machine Learning."
    else:
        with open(TXT_FILE, "r", encoding="utf-8") as f:
            text_data = f.read()

    # Split text into paragraphs
    chunks = [c.strip() for c in text_data.split('\n\n') if c.strip()]
    knowledge_base = chunks
    print(f"Loaded {len(chunks)} chunks of Hopsworks info.")
    
    if chunks:
        knowledge_embeddings = embedder.encode(chunks, convert_to_tensor=False)
        knowledge_embeddings = knowledge_embeddings / np.linalg.norm(knowledge_embeddings, axis=1, keepdims=True)

load_knowledge_base()

def retrieve_info(query, top_k=2):
    """Finds the most relevant info about Hopsworks"""
    if knowledge_embeddings is None or len(knowledge_base) == 0:
        return ""
    
    query_emb = embedder.encode([query])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    
    scores = np.dot(knowledge_embeddings, query_emb.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    retrieved_text = "\n\n".join([knowledge_base[i] for i in top_indices])
    return retrieved_text

# -------------------------------------------------------------
# 3. CHAT RESPONSE LOGIC
# -------------------------------------------------------------
def respond(message, history):
    
    # A. Retrieve context
    context = retrieve_info(message)
    
    # B. System Prompt: Hopsworks Expert Persona
    system_prompt = f"""
You are a senior MLOps engineer and an expert on the Hopsworks platform.
Answer the user's question using ONLY the context provided below.
If the context doesn't contain the answer, say "I don't have that information in my Hopsworks knowledge base."
Be technical but clear.

CONTEXT:
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]
    
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    response_iter = llm.create_chat_completion(
        messages=messages,
        stream=True,
        max_tokens=512,
        temperature=0.5, # Low temp for technical accuracy
    )
    
    partial_message = ""
    for chunk in response_iter:
        if "content" in chunk["choices"][0]["delta"]:
            token = chunk["choices"][0]["delta"]["content"]
            partial_message += token
            yield partial_message

# -------------------------------------------------------------
# 4. USER INTERFACE
# -------------------------------------------------------------
demo = gr.ChatInterface(
    fn=respond,
    title="🟢 Hopsworks Expert AI",
    description=f"Ask me anything about the Hopsworks Feature Store. Powered by RAG + Llama-3B.",
    examples=[
        "What is Hopsworks?", 
        "How do I connect to the Feature Store?", 
        "Can I use it for real-time inference?"
    ],
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()