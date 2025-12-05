import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Hopsworks Expert AI", page_icon="🟢")
st.title("🟢 Hopsworks Expert AI")
st.caption("RAG System powered by Llama-3B & Hopsworks Docs")

# -------------------------------------------------------------
# 1. FUNZIONI DI CARICAMENTO (CACHED)
# -------------------------------------------------------------
# @st.cache_resource è FONDAMENTALE in Streamlit:
# impedisce di riscaricare il modello a ogni messaggio.

@st.cache_resource
def load_llm():
    REPO_ID = "abertekth/model"
    FILENAME = "mio-modello-q4_k_m.gguf"

    print(f"--- Downloading LLM: {FILENAME} ---")
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.stop()

    print("--- Loading Llama into RAM ---")
    # Carichiamo il modello
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,         
        n_threads=2,        
        n_gpu_layers=0,     # CPU only
        verbose=False
    )
    return llm

@st.cache_resource
def load_rag_system():
    print("--- Preparing RAG System ---")
    # IMPORTANTE: device='cpu' per evitare errore Meta Tensor su Cloud
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    
    TXT_FILE = "knowledge.txt"
    knowledge_base = []
    knowledge_embeddings = None

    if not os.path.exists(TXT_FILE):
        # Dati dummy se manca il file
        text_data = "Hopsworks is a Feature Store."
    else:
        with open(TXT_FILE, "r", encoding="utf-8") as f:
            text_data = f.read()

    # Creiamo i chunk
    chunks = [c.strip() for c in text_data.split('\n\n') if c.strip()]
    
    if chunks:
        # Creiamo gli embeddings
        knowledge_embeddings = embedder.encode(chunks, convert_to_tensor=False)
        knowledge_embeddings = knowledge_embeddings / np.linalg.norm(knowledge_embeddings, axis=1, keepdims=True)
    
    return embedder, chunks, knowledge_embeddings

# --- INIZIALIZZAZIONE ---
try:
    with st.spinner("Caricamento del cervello AI in corso..."):
        llm = load_llm()
        embedder, knowledge_base, knowledge_embeddings = load_rag_system()
except Exception as e:
    st.error(f"Errore critico all'avvio: {e}")
    st.stop()

# -------------------------------------------------------------
# 2. LOGICA RAG (RICERCA)
# -------------------------------------------------------------
def retrieve_info(query, top_k=2):
    if knowledge_embeddings is None or len(knowledge_base) == 0:
        return ""
    
    # Vettorizza la domanda
    query_emb = embedder.encode([query])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    
    # Calcola similarità
    scores = np.dot(knowledge_embeddings, query_emb.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    retrieved_text = "\n\n".join([knowledge_base[i] for i in top_indices])
    return retrieved_text

# -------------------------------------------------------------
# 3. INTERFACCIA CHAT (STREAMLIT STYLE)
# -------------------------------------------------------------

# Inizializza la storia se non esiste
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful Hopsworks expert."}
    ]

# Mostra i messaggi precedenti (tranne il system prompt)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input Utente
if prompt := st.chat_input("Ask about Hopsworks..."):
    
    # 1. Mostra messaggio utente
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Recupera Contesto RAG
    context = retrieve_info(prompt)
    
    # 3. Prepara il System Prompt Dinamico
    system_prompt_rag = f"""
You are a senior MLOps engineer and an expert on the Hopsworks platform.
Answer the user's question using ONLY the context provided below.
If the context doesn't contain the answer, say "I don't have that information in my Hopsworks knowledge base."
Be technical but clear.

CONTEXT:
{context}
"""
    
    # Creiamo una lista temporanea di messaggi per l'LLM (