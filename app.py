import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# -------------------------------------------------------------
# CONFIGURAZIONE MODELLO
# -------------------------------------------------------------
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
    n_gpu_layers=0,       # CPU-only (Spaces free tier)
    n_ctx=2048,          # Context window
    n_threads=2,         # Only 2 vCPUs available
    verbose=False
)
print("Model loaded successfully!")


# -------------------------------------------------------------
# FUNZIONE DI RISPOSTA DEL MODELLO
# -------------------------------------------------------------
def respond(message, history):
    """
    Funzione chiamata da Gradio che costruisce il prompt,
    invia il contesto al modello e streamma la risposta.
    """

    system_prompt = """
You are the Game Master of a word-guessing game.

1. At the start of the game, secretly choose one word in English.
   - The word must be a common noun.
   - Do not reveal the word unless the player guesses correctly.

2. The player can:
   - Ask yes/no questions
   - Ask for hints
   - Make guesses

3. Your rules:
   - Always answer truthfully.
   - Never reveal the hidden word before the correct guess.
   - If the player guesses the word exactly, reply:
       "Correct! The hidden word was: <word>"
       Then immediately choose a new hidden word (keep it secret).

4. Keep it fun:
   - Keep answers short.
   - Give subtle hints if the player is stuck.
   - Never break character.

Begin the game by saying:
"Welcome to *Guess the Hidden Word*! I have chosen a word. Ask me anything or make a guess!"
"""

    # Costruzione messaggi per il modello
    messages = [{"role": "system", "content": system_prompt}]

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    # Generazione risposta in streaming
    response_iter = llm.create_chat_completion(
        messages=messages,
        stream=True,
        max_tokens=256,
        temperature=0.7
    )

    partial = ""
    for chunk in response_iter:
        if "content" in chunk["choices"][0]["delta"]:
            token = chunk["choices"][0]["delta"]["content"]
            partial += token
            yield partial


# -------------------------------------------------------------
# INTERFACCIA GRADIO
# -------------------------------------------------------------
demo = gr.ChatInterface(
    fn=respond,
    title="Guess The Hidden Word - Llama GGUF",
    description="A word-guessing mini-game powered by your fine-tuned Llama-3.1-8B model.",
    examples=[
        "Is it an animal?",
        "Is it something you can eat?",
        "Give me a subtle hint."
    ],
    cache_examples=False
)


# -------------------------------------------------------------
# AVVIO APP
# -------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()