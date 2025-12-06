# 🦢 Mac64 RAG - Local LLM Chatbot with Streamlit

**Mac64 RAG** is a lightweight Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, and **Llama.cpp**. It allows users to upload documents (PDF or TXT) and chat with them using a quantized Llama-3.2 model running locally on the CPU.

## 🚀 Key Features

  * **RAG Architecture:** retrieval of relevant document chunks to ground the LLM's answers.
  * **Local Inference:** Uses `llama-cpp-python` to run GGUF quantized models efficiently on CPU.
  * **Document Support:** Upload and parse `.pdf` and `.txt` files.
  * **Vector Search:** Utilizes **FAISS** for fast similarity search and **Sentence Transformers** for embeddings.
  * **Strict Context Mode:** The system prompt is engineered to answer *only* based on retrieved context, reducing hallucinations.

-----

## 🛠️ Tech Stack

  * **UI:** [Streamlit](https://streamlit.io/)
  * **LLM Runtime:** [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
  * **Orchestration:** [LangChain](https://www.langchain.com/)
  * **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
  * **Vector Store:** FAISS (CPU)
  * **Model:** Llama-3.2 3B Quantized (`mio-modello-q4_k_m.gguf`) downloaded from Hugging Face (`abertekth/model`).

-----

## 📦 Installation

### Prerequisites

To run `llama-cpp-python`, your system requires C++ build tools.

  * **Debian/Ubuntu/Streamlit Cloud:**
    You must install `build-essential` and `cmake.
    ```bash
    sudo apt-get update
    sudo apt-get install build-essential cmake
    ```

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### Step 2: Set up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Step 3: Install Python Dependencies

The application relies on specific versions of `llama-cpp-python` optimized for AVX2 instructions to ensure efficient loading and inference.

Run the following command to install all requirements:

```bash
pip install -r requirements.txt
```

**Note on `requirements.txt` content:**
The `requirements.txt` file includes a specific extra index URL for the `llama-cpp-python` wheel:

```text
--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/avx2
llama-cpp-python==0.2.90
```

-----

## 🏃‍♂️ Usage

1.  **Start the Streamlit App:**

    ```bash
    streamlit run streamlit.py
    ```

2.  **Model Download:**
    On the first run, the app will automatically download the quantized model (`mio-modello-q4_k_m.gguf`) from Hugging Face. This may take a few minutes depending on your connection.

3.  **Upload Documents:**

      * Open the Sidebar on the left.
      * Upload a PDF or TXT file.
      * Wait for the "Indexing of the documents in progress..." spinner to finish.

4.  **Chat:**

      * Type your question in the chat input.
      * The model will retrieve the top 3 relevant chunks from your document and generate an answer based **strictly** on that context.

-----

## ⚙️ Configuration

The application is configured via constants in `streamlit.py`:

  * **`REPO_ID`**: `abertekth/model` - The Hugging Face repository source.
  * **`FILENAME`**: `mio-modello-q4_k_m.gguf` - The specific model file.
  * **`EMBEDDING_MODEL`**: `sentence-transformers/all-MiniLM-L6-v2` - A lightweight embedding model suitable for CPU usage.
  * **Context Window**: set to `2048` tokens with `n_threads=2` to ensure stability on limited hardware (like Streamlit Cloud free tier).
