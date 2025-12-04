
import streamlit as st
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# --- CONFIGURAZIONE ---
REPO_ID = "ehsisi/lab2_gguf" 
MODEL_FILENAME = "model-3b-Q8_0.gguf" # Es: unsloth.Q4_K_M.gguf

st.set_page_config(page_title="Lab 2: Fast Inference", page_icon="⚡")

@st.cache_resource
def load_model():
    # Scarica il modello
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    
    # Carica con ctransformers (Niente compilazione!)
    # Nota: model_type="llama" funziona anche per Llama-3 nella maggior parte dei casi
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="llama",
        context_length=2048,
        gpu_layers=0 # Forza CPU
    )
    return llm

st.title("⚡ ID2223 Lab 2: CTransformers Edition")

try:
    with st.spinner("Caricamento modello... (sarà veloce!)"):
        llm = load_model()
    st.success("Modello pronto.")
except Exception as e:
    st.error(f"Errore: {e}")

prompt = st.text_area("Inserisci prompt:")

if st.button("Genera"):
    if prompt:
        # Template manuale per Llama 3
        full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
        
        with st.spinner("Generazione..."):
            # ctransformers genera testo direttamente
            response = llm(full_prompt, max_new_tokens=256)
            
            st.write(response)