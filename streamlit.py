import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

import tempfile

# --- LIBRERIE RAG ---
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURAZIONE ---
REPO_ID = "abertekth/model"
FILENAME = "mio-modello-q4_k_m.gguf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Modello leggero per embeddings
# ----------------------

st.set_page_config(page_title="Abertek AI Cloud", page_icon="☁️")

st.title("☁️ Abertek AI - Streamlit Cloud")
st.caption("Modello Llama-3.2 3B Quantizzato (GGUF)")

# --- CARICAMENTO MODELLO LLM ---
@st.cache_resource
def load_model():
    # Mostra uno spinner mentre scarica
    with st.spinner(f'Scaricamento modello da Hugging Face ({FILENAME})... Attendere...'):
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    # Carica in RAM
    # NOTA: Su Streamlit Cloud la CPU è limitata, usiamo thread bassi
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context window
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

# --- CARICAMENTO MODELLO EMBEDDING ---
@st.cache_resource
def load_embedding_model():
    with st.spinner('Caricamento modello embeddings...'):
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

try:
    llm = load_model()
    embeddings = load_embedding_model()
except Exception as e:
    st.error(f"Errore caricamento modelli: {e}")
    st.stop()

# --- SIDEBAR: GESTIONE DOCUMENTI ---
with st.sidebar:
    st.header("📂 Carica Documenti")
    uploaded_file = st.file_uploader("Carica un file .txt o .pdf", type=["txt", "pdf"])
    
    if uploaded_file and "vector_store" not in st.session_state:
        with st.spinner("Indicizzazione documento in corso..."):
            try:
                # Salva file temporaneo
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Carica in base al tipo
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_file_path)
                else:
                    loader = TextLoader(tmp_file_path, encoding="utf-8")
                
                docs = loader.load()
                
                # Split del testo in chunk
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, # Chunk piccoli per risparmiare contesto
                    chunk_overlap=50
                )
                chunks = text_splitter.split_documents(docs)
                
                # Creazione Vector Store (FAISS)
                vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.vector_store = vector_store
                st.success(f"Indicizzato {len(chunks)} porzioni di testo!")
                
                # Pulizia file temp
                os.remove(tmp_file_path)
                
            except Exception as e:
                st.error(f"Errore lettura file: {e}")

    if st.button("Reset Chat e Memoria"):
        st.session_state.messages = [{"role": "system", "content": "Sei un assistente utile."}]
        if "vector_store" in st.session_state:
            del st.session_state.vector_store
        st.rerun()

# --- INTERFACCIA CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Sei un assistente utile che risponde basandosi sul contesto fornito."}
    ]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Scrivi qui..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- LOGICA RAG ---
    context_text = ""
    if "vector_store" in st.session_state:
        # 1. Cerca i pezzi più rilevanti (k=3 per non saturare la RAM/Contesto)
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(prompt)
        
        # 2. Costruisci il contesto
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 3. Prompt Augmentato
        augmented_prompt = f"""Usa il seguente contesto per rispondere alla domanda. Se non sai la risposta, dillo chiaramente.
        
        CONTESTO:
        {context_text}
        
        DOMANDA:
        {prompt}
        """
        
        # Sostituiamo l'ultimo messaggio utente con quello aumentato per l'LLM (ma non nell'UI)
        messages_for_llm = st.session_state.messages[:-1] + [{"role": "user", "content": augmented_prompt}]
    else:
        # Nessun documento caricato, usa chat normale
        messages_for_llm = st.session_state.messages

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Generazione
        stream = llm.create_chat_completion(
            messages=messages_for_llm,
            stream=True,
            max_tokens=512,
            temperature=0.7
        )
        
        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                full_response += chunk["choices"][0]["delta"]["content"]
                message_placeholder.markdown(full_response + "▌")
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})