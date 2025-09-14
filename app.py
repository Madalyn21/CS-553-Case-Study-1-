import gradio as gr
from huggingface_hub import InferenceClient
import os
import json
import random

pipe = None

# Load WPI facts from JSON
with open("facts.json", "r") as f:
    WPI_FACTS = json.load(f)

# Fancy CSS (same as before)
fancy_css = """ ... """  # Keep your previous CSS here

# --- Gompei chatbot response ---
def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    use_local_model: bool,
    hf_token: gr.OAuthToken = None,
):
    global pipe

    # Pick a random WPI fact
    fact = random.choice(WPI_FACTS)["text"]

    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    # Append the user message + random fact for context
    messages.append({"role": "user", "content": f"{message}\n\nFun fact: {fact}"})

    response = ""

    if use_local_model:
        print("[MODE] local")
        from transformers import pipeline
        if pipe is None:
            pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")

        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        response = outputs[0]["generated_text"][len(prompt):]
        yield response.strip()

    else:
        print("[MODE] api")
        token_value = None
        if hf_token and getattr(hf_token, "token", None):
            token_value = hf_token.token
        elif os.environ.get("HF_TOKEN"):
            token_value = os.environ.get("HF_TOKEN")

        if not token_value:
            yield "⚠️ Please log in with your Hugging Face account or set HF_TOKEN in environment."
            return

        client = InferenceClient(token=os.environ["HF_TOKEN"], model="openai/gpt-oss-20b")

        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            choices = chunk.choices
            token = ""
            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content
            response += token
            yield response

# --- Chat Interface ---
chatbot = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(
            value="You are Gompei the Goat, WPI's mascot. Answer questions with fun goat-like personality and real WPI facts. Keep the responses short",
            label="System message",
        ),
        gr.Slider(minimum=1, maximum=1024, value=256, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.Checkbox(label="Use Local Model", value=False),
    ],
    type="messages",
    examples=[
        [
            "Where is WPI located?",
            "You are Gompei the Goat, WPI's mascot. Answer questions with fun goat-like personality and real WPI facts.",
            128,
            0.7,
            0.95,
            False
        ],
        [
            "Who founded WPI?",
            "You are Gompei the Goat, WPI's mascot. Answer questions with fun goat-like personality and real WPI facts.",
            128,
            0.7,
            0.95,
            False
        ],
    ],
)

# --- Blocks layout ---
with gr.Blocks(css=fancy_css) as demo:
    with gr.Row():
        gr.Markdown("<h1 id='title'>🐐 Chat with Gompei</h1>")
        gr.LoginButton()
    chatbot.render()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)