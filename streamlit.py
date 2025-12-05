import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

# --- CONFIGURAZIONE ---
REPO_ID = "abertekth/model"
FILENAME = "mio-modello-q4_k_m.gguf"
# ----------------------

st.set_page_config(page_title="Abertek AI Cloud", page_icon="☁️")

st.title("☁️ Abertek AI - Streamlit Cloud")
st.caption("Modello Llama-3.2 3B Quantizzato (GGUF)")

@st.cache_resource
def load_model():
    # Mostra uno spinner mentre scarica
    with st.spinner(f'Scaricamento modello da Hugging Face ({FILENAME})... Attendere...'):
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    # Carica in RAM
    # NOTA: Su Streamlit Cloud la CPU è limitata, usiamo thread bassi
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=2, 
        verbose=False
    )
    return llm

# Gestione errori di memoria (Streamlit Cloud Free ha poca RAM)
try:
    llm = load_model()
except Exception as e:
    st.error(f"Errore: Potresti aver superato la memoria di Streamlit Cloud. Dettagli: {e}")
    st.stop()

# --- INTERFACCIA CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Sei un assistente utile."}
    ]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Scrivi qui..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        stream = llm.create_chat_completion(
            messages=st.session_state.messages,
            stream=True,
            max_tokens=512,
            temperature=0.7
        )
        
        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                full_response += chunk["choices"][0]["delta"]["content"]
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})