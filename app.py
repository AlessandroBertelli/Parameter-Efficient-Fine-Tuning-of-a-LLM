import streamlit as st
import os
import requests
from llama_cpp import Llama

# --- CONFIGURAZIONE ---
# Sostituisci con i tuoi dati esatti
REPO_ID = "ehsisi/lab2_gguf"
MODEL_FILENAME = "model-3b-Q8_0.gguf"

# URL diretto per il download da Hugging Face
MODEL_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/{MODEL_FILENAME}"

st.set_page_config(page_title="Lab 2: LLM Monitor", page_icon="📊")

def download_model_with_progress(url, dest_path):
    """
    Scarica il modello mostrando una barra di progresso nell'UI.
    """
    if os.path.exists(dest_path):
        st.info(f"Modello già presente: {dest_path}")
        return

    st.write(f"Inizio download da: {url}")
    
    # Avvia la richiesta streaming
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Creiamo gli elementi grafici
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    downloaded_size = 0
    chunk_size = 1024 * 1024  # Scarica 1MB alla volta
    
    with open(dest_path, "wb") as file:
        for data in response.iter_content(chunk_size):
            file.write(data)
            downloaded_size += len(data)
            
            # Calcolo percentuale
            if total_size > 0:
                percent = downloaded_size / total_size
                # Streamlit accetta valori tra 0.0 e 1.0
                progress_bar.progress(min(percent, 1.0))
                
                # Aggiorna il testo (es. "350 / 3400 MB")
                mb_downloaded = downloaded_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                status_text.text(f"Scaricamento: {mb_downloaded:.1f} MB / {mb_total:.1f} MB")

    status_text.success("Download completato!")
    progress_bar.empty() # Rimuove la barra alla fine

@st.cache_resource
def load_llm():
    model_local_path = MODEL_FILENAME
    
    # 1. Scarica con progresso visibile
    download_model_with_progress(MODEL_URL, model_local_path)
    
    # 2. Carica il modello in RAM
    print("Caricamento Llama in memoria...")
    llm = Llama(
        model_path=model_local_path,
        n_ctx=512,        # <--- RIDOTTO PER EVITARE CRASH
        n_threads=2,
        verbose=True
    )
    return llm

st.title("🤖 Lab 2: Download Monitor")

# Avvia il processo
try:
    llm = load_llm()
    st.success("Modello Caricato e Pronto!")
    
    # Interfaccia Chat
    prompt = st.text_area("Scrivi il tuo prompt:")
    if st.button("Genera"):
        if prompt:
            full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
            with st.spinner("Generazione risposta..."):
                output = llm(full_prompt, max_tokens=200, stop=["<|end|>"], echo=False)
                st.write(output['choices'][0]['text'])

except Exception as e:
    st.error(f"Errore: {e}")