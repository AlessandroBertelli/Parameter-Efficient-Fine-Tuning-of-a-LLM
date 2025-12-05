import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

REPO_ID = "abertekth/model"
FILENAME = "mio-modello-q4_k_m.gguf"

print(f"Downloading {FILENAME} from {REPO_ID}...")
model_path = hf_hub_download(
    repo_id=REPO_ID, 
    filename=FILENAME
)
print(f"Model downloaded to: {model_path}")

llm = Llama(
    model_path=model_path,
    n_gpu_layers=0,        # Use 0 for CPU-only Spaces
    n_ctx=4096,            # Context window
    n_threads=2,           # Align with the 2 vCPUs on the free tier
    verbose=False
)
print("Model loaded successfully!")

def chat_response(message, history):
    
    system_prompt = "You are a helpful and detailed assistant. Your name is José."
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
        
    messages.append({"role": "user", "content": message})

    # Generate response
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.7,
        stream=True  
    )
    
    partial_message = ""
    for chunk in response:
        if 'content' in chunk['choices'][0]['delta']:
            text_chunk = chunk['choices'][0]['delta']['content']
            partial_message += text_chunk
            yield partial_message

demo = gr.ChatInterface(
    fn=chat_response,
    title="ID2223 FineLlama Chat (GGUF)",
    description="Chat with the fine-tuned Llama-3.1-8B model running on CPU via llama.cpp.",
    examples=["Tell me about ID2223.", "Who are you?", "Explain quantization."],
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()