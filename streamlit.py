import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os
import tempfile

# --- RAG libraries ---
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------
REPO_ID = "abertekth/model"
FILENAME = "mio-modello-q4_k_m.gguf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Lightweight model for embeddings
system_content = """You are a Retrieval-Augmented Generation (RAG) assistant.
Your answers must be based solely and strictly on the information contained in the retrieved documents provided in the context.
Rules:
1. Do not use any outside knowledge, assumptions, or facts not explicitly present in the retrieved context.
2. If the answer is not directly supported by the retrieved documents, reply with: "The provided documents do not contain enough information to answer this question."
3. When relevant, cite the specific document sections you are using.
4. Do not invent details, do not guess, and do not fill gaps with general world knowledge.
5. If the user asks for information that contradicts the documents, clarify that the documents do not support that claim.
6. Your goal is to provide accurate, context-grounded answers using only the retrieved sources."""


st.set_page_config(page_title="Mac64 RAG", page_icon="🦢")
st.title("🦢 Mac64 RAG - Streamlit Cloud")
st.caption("Llama-3.2 3B Quantized (GGUF Model)")

# --------------------------------------------------------------
# LLM LOADING
# --------------------------------------------------------------
@st.cache_resource
def load_model():
    # Show a spinner while downloading
    with st.spinner(f'Download the model from Hugging Face ({FILENAME})... Wait...'):
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    # Load in RAM
    # NOTE: Few threads due to Streamlit Cloud CPU limits
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context window
        n_threads=2,  # number of threads
        verbose=False
    )
    return llm

# Memory error handling (Streamlit Cloud Free has little RAM)
try:
    llm = load_model()
except Exception as e:
    st.error(f"Error: you could exceed the memory of Streamlit Cloud. Details: {e}")
    st.stop()

# --------------------------------------------------------------
# EMBEDDING MODEL LOADING
# Used to convert the uploaded docuemnts and user queries into embeddings.
# We use here a lightweight model to save RAM.
# --------------------------------------------------------------
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

# --------------------------------------------------------------
# SIDEBAR - UPLOAD DOCUMENTS
# --------------------------------------------------------------
with st.sidebar:
    st.header("📂 Upload documents here")
    uploaded_file = st.file_uploader("Upload a file .txt o .pdf", type=["txt", "pdf"])
    
    if uploaded_file and "vector_store" not in st.session_state:
        with st.spinner("Indexing of the documents in progress..."):
            try:
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Upload the file based on its type
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_file_path)
                else:
                    loader = TextLoader(tmp_file_path, encoding="utf-8")
                
                docs = loader.load()
                
                # Slitting to small chunks to save context
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, # Small chunks to save context
                    chunk_overlap=50
                )
                chunks = text_splitter.split_documents(docs)
                
                # Create Vector Store (FAISS). Lightweight vector store for Streamlit Cloud
                vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.vector_store = vector_store
                st.success(f"Indexed {len(chunks)} portions of text!")
                
                # Clean up temporary file
                os.remove(tmp_file_path)
                
            except Exception as e:
                st.error(f"Errore lettura file: {e}")

    if st.button("Reset Chat and Memory"):
        st.session_state.messages = [{"role": "system", "content": system_content}]
        if "vector_store" in st.session_state:
            del st.session_state.vector_store
        st.rerun()


# --------------------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------------------
# Give the system content only once at the beginning
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_content}
    ]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Prompt input
if prompt := st.chat_input("Write here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ------------------------------------------
    # RAG LOGIC
    # ------------------------------------------
    context_text = ""
    if "vector_store" in st.session_state:
        # Search for the most relevant pieces (k=3 to avoid saturating RAM/Context)
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(prompt)
        
        # Build the context
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # Create the augmented prompt to answer using the context only
        augmented_prompt = f"""Use ONLY the following context to answer the question.
        
        CONTEXT:
        {context_text}
        
        QUESTION:
        {prompt}
        """
        
        # Replace the last user message with the augmented one for the LLM (but not in the UI)
        messages_for_llm = st.session_state.messages[:-1] + [{"role": "user", "content": augmented_prompt}]
    else:
        # Use the normal chat because no document is uploaded
        messages_for_llm = st.session_state.messages

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Generate the response in streaming mode
        stream = llm.create_chat_completion(
            messages=messages_for_llm,
            stream=True,  # Enable streaming mode
            max_tokens=512,  # Response length limit
            temperature=0.7  # Creativity control
        )
        
        # Stream the response and update the message placeholder
        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                full_response += chunk["choices"][0]["delta"]["content"]
                message_placeholder.markdown(full_response + "▌")
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})