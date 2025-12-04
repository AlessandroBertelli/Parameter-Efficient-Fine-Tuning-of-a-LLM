import streamlit as st
import os
import requests
from ctransformers import AutoModelForCausalLM # <--- Usiamo questa, non Llama

# --- CONFIGURAZIONE ---
REPO_ID = "ehsisi/lab2_gguf" # <--- INSERISCI I TUOI DATI
MODEL_FILENAME = "model-3b-Q8_0.gguf"    # <--- INSERISCI I TUOI DATI

# URL diretto per il download
MODEL_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/{MODEL_FILENAME}"

st.set_page_config(page_title="Lab 2: Fast Inference", page_icon="⚡")

def download_model_with_progress(url, dest_path):
    """
    Scarica il modello mostrando una barra di progresso.
    """
    if os.path.exists(dest_path):
        st.info(f"✅ Modello trovato in cache: {dest_path}")
        return

    st.write(f"⏳ Inizio download da Hugging Face...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    downloaded_size = 0
    chunk_size = 1024 * 1024 # 1MB chunks
    
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

@st.cache_resource
def load_model():
    model_local_path = MODEL_FILENAME
    
    # 1. Scarica (se non c'è già)
    download_model_with_progress(MODEL_URL, model_local_path)
    
    # 2. Carica con CTransformers (Veloce, niente compilazione)
    print("Inizializzazione CTransformers...")
    llm = AutoModelForCausalLM.from_pretrained(
        model_local_path,
        model_type="llama",
        context_length=512, # Basso per salvare RAM
        gpu_layers=0        # Forza CPU
    )
    return llm

st.title("⚡ Lab 2: CTransformers UI")

try:
    # Mostra lo spinner mentre scarica/carica
    llm = load_model()
    st.success("Motore AI pronto all'uso!")

    prompt = st.text_area("Inserisci il tuo prompt:")
    
    if st.button("Genera"):
        if prompt:
            full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
            with st.spinner("Generazione in corso..."):
                # CTransformers usa una sintassi semplice
                response = llm(full_prompt, max_new_tokens=200)
                st.write(response)

except Exception as e:
    st.error(f"Errore critico: {e}")