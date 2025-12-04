import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# --- CONFIGURAZIONE ---
# Inserisci qui il tuo Repo ID di Hugging Face e il nome del file GGUF
REPO_ID = "IL_TUO_USERNAME/IL_TUO_REPO_MODEL" 
MODEL_FILENAME = "nome_del_tuo_modello.gguf" # Es: unsloth.Q4_K_M.gguf

st.set_page_config(page_title="Lab 2: LLM Inference", page_icon="🤖")

@st.cache_resource
def load_model():
    """
    Scarica e cachea il modello.
    Questa funzione viene eseguita solo una volta all'avvio.
    """
    print(f"Scaricamento del modello {MODEL_FILENAME} da Hugging Face...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    
    print("Inizializzazione Llama...")
    # n_ctx definisce la lunghezza del contesto (input + output). 
    # n_threads usa i core della CPU disponibili.
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,      
        n_threads=2      
    )
    return llm

# --- INTERFACCIA UTENTE ---
st.title("🤖 ID2223 Lab 2: Fine-Tuned LLM")
st.markdown("Questo modello è stato fine-tunato e convertito in GGUF per l'inferenza su CPU.")

# Caricamento del modello (con spinner visivo)
try:
    with st.spinner("Caricamento del modello in corso... (richiede circa 1 minuto la prima volta)"):
        llm = load_model()
    st.success("Modello caricato e pronto!")
except Exception as e:
    st.error(f"Errore nel caricamento del modello: {e}")

# Input utente
prompt = st.text_area("Inserisci la tua richiesta:", height=100)

if st.button("Genera Risposta"):
    if prompt:
        with st.spinner("Generazione in corso..."):
            # FORMATTAZIONE DEL PROMPT
            # Nota: Llama-3 usa un formato specifico. Se usi un altro modello, adatta questo template.
            # Questo è il formato standard ChatML/Llama-3
            full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
            
            # INFERENZA
            output = llm(
                full_prompt,
                max_tokens=256,  # Limita la lunghezza della risposta
                stop=["<|end|>", "<|user|>"], # Token di stop per evitare che il modello parli da solo
                echo=False,
                temperature=0.7
            )
            
            # Estrazione del testo
            response_text = output['choices'][0]['text']
            
            st.markdown("### Risposta:")
            st.write(response_text)
    else:
        st.warning("Per favore scrivi qualcosa prima di generare.")

# Footer per soddisfare i requisiti del lab
st.markdown("---")
st.caption("Lab 2 - ID2223 | Fine-Tuning & Deployment")