import streamlit as st
import os
import requests
from llama_cpp import Llama  # <--- Torniamo alla libreria ufficiale

# --- CONFIGURAZIONE ---
# Inserisci il tuo REPO_ID esatto qui sotto
REPO_ID = "abertekth/GGUF-Lab2" 
# Inserisci il nome del file esatto che hai citato nell'errore
MODEL_FILENAME = "model-unsloth.Q4_K_M.gguf"

MODEL_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/{MODEL_FILENAME}"

st.set_page_config(page_title="Lab 2: Llama CPP", page_icon="🦙")

def download_model_with_progress(url, dest_path):
    if os.path.exists(dest_path):
        st.info(f"✅ Modello trovato: {dest_path}")
        return

    st.write(f"⏳ Inizio download da Hugging Face...")
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded_size = 0
        chunk_size = 1024 * 1024 # 1MB
        
        with open(dest_path, "wb") as file:
            for data in response.iter_content(chunk_size):
                file.write(data)
                downloaded_size += len(data)
                if total_size > 0:
                    percent = downloaded_size / total_size
                    progress_bar.progress(min(percent, 1.0))
                    mb_cur = downloaded_size / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    status_text.text(f"Scaricamento: {mb_cur:.1f} MB / {mb_tot:.1f} MB")
        
        status_text.success("Download completato!")
        progress_bar.empty()
    except Exception as e:
        st.error(f"Errore durante il download: {e}")

@st.cache_resource
def load_model():
    model_local_path = MODEL_FILENAME
    
    # 1. Scarica
    download_model_with_progress(MODEL_URL, model_local_path)
    
    # 2. Carica con llama-cpp-python
    # Questo funziona con tutti i GGUF moderni
    print("Inizializzazione Llama...")
    llm = Llama(
        model_path=model_local_path,
        n_ctx=512,        # Manteniamo basso per la RAM
        n_threads=2,      # CPU cores
        verbose=True
    )
    return llm

st.title("🦙 Lab 2: 3B Model UI")

try:
    llm = load_model()
    st.success("Modello caricato con successo!")

    prompt = st.text_area("Scrivi il tuo prompt:")
    
    if st.button("Genera"):
        if prompt:
            # Template standard
            full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
            with st.spinner("Generazione..."):
                output = llm(
                    full_prompt, 
                    max_tokens=256, 
                    stop=["<|end|>"], 
                    echo=False
                )
                st.write(output['choices'][0]['text'])

except Exception as e:
    st.error(f"Errore critico: {e}")