import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

import tempfile

# --- LIBRERIE RAG ---
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURAZIONE ---
REPO_ID = "abertekth/model"
FILENAME = "mio-modello-q4_k_m.gguf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Modello leggero per embeddings
# ----------------------

st.set_page_config(page_title="RAG AI Cloud", page_icon="☁️")

st.title("☁️ RAG - Streamlit Cloud")
st.caption("Llama-3.2 3B Quantizzato (GGUF)")

# --- CARICAMENTO MODELLO LLM ---
@st.cache_resource
def load_model():
    # Mostra uno spinner mentre scarica
    with st.spinner(f'Download the model from Hugging Face ({FILENAME})... Wait...'):
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
    st.error(f"Error: you could exceed the memory of Streamlit Cloud. Details: {e}")
    st.stop()

# --- CARICAMENTO MODELLO EMBEDDING ---
@st.cache_resource
def load_embedding_model():
    with st.spinner('Loading the embedding model...'):
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

try:
    llm = load_model()
    embeddings = load_embedding_model()
except Exception as e:
    st.error(f"Error to load models: {e}")
    st.stop()

# --- SIDEBAR: GESTIONE DOCUMENTI ---
with st.sidebar:
    st.header("📂 Upload documents here")
    uploaded_file = st.file_uploader("Upload a file .txt o .pdf", type=["txt", "pdf"])
    
    if uploaded_file and "vector_store" not in st.session_state:
        with st.spinner("Indexing of the documents in progress..."):
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
                st.success(f"Indexed {len(chunks)} portions of text!")
                
                # Pulizia file temp
                os.remove(tmp_file_path)
                
            except Exception as e:
                st.error(f"Errore lettura file: {e}")

    if st.button("Reset Chat and Memory"):
        st.session_state.messages = [{"role": "system", "content": "You are a really useful assistant."}]
        if "vector_store" in st.session_state:
            del st.session_state.vector_store
        st.rerun()

# --- INTERFACCIA CHAT ---
system_content = """You are a Retrieval-Augmented Generation (RAG) assistant.
Your answers must be based solely and strictly on the information contained in the retrieved documents provided in the context.
Rules:
1. Do not use any outside knowledge, assumptions, or facts not explicitly present in the retrieved context.
2. If the answer is not directly supported by the retrieved documents, reply with: "The provided documents do not contain enough information to answer this question."
3. When relevant, cite the specific document sections you are using.
4. Do not invent details, do not guess, and do not fill gaps with general world knowledge.
5. If the user asks for information that contradicts the documents, clarify that the documents do not support that claim.
6. Your goal is to provide accurate, context-grounded answers using only the retrieved sources."""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_content}
    ]


# --- WhatsApp-style chat CSS and rendering ---
st.markdown("""
<style>
.whatsapp-chat {
    max-width: 600px;
    margin: 0 auto;
    padding: 10px;
}
.bubble {
    padding: 10px 16px;
    border-radius: 18px;
    margin-bottom: 8px;
    display: inline-block;
    max-width: 80%;
    font-size: 16px;
    position: relative;
    word-break: break-word;
}
.bubble.user {
    background: #dcf8c6;
    color: #222;
    align-self: flex-end;
    float: right;
    clear: both;
}
.bubble.assistant {
    background: #fff;
    color: #222;
    border: 1px solid #ececec;
    align-self: flex-start;
    float: left;
    clear: both;
}
.bubble.system {
    background: #e9e9e9;
    color: #888;
    text-align: center;
    margin: 0 auto 12px auto;
    display: block;
    border-radius: 12px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="whatsapp-chat">', unsafe_allow_html=True)
for message in st.session_state.messages:
    role = message["role"]
    if role == "system":
        st.markdown(f'<div class="bubble system">{message["content"]}</div>', unsafe_allow_html=True)
    elif role == "user":
        st.markdown(f'<div class="bubble user">{message["content"]}</div>', unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f'<div class="bubble assistant">{message["content"]}</div>', unsafe_allow_html=True)
    # Add more roles if needed
st.markdown('</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Write here..."):
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
        augmented_prompt = f"""Use ONLY the following context to answer the question.
        
        CONTEXT:
        {context_text}
        
        QUESTION:
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