from pathlib import Path
import time
import streamlit as st
from chatbot_runtime import load_chatbot, generate_response

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "artifacts" / "chatbot_transformer_best.pt"

st.set_page_config(page_title="Cornell Dialogue AI", page_icon="💬", layout="centered")
st.title("Intelligent Conversational Chatbot")
st.caption("Custom PyTorch encoder–decoder Transformer trained on the Cornell Movie Dialogs Corpus.")
st.info("This is an experimental dialogue model, not a factual assistant. Responses can be repetitive, irrelevant, or inappropriate because the training corpus contains movie dialogue.")

@st.cache_resource
def cached_bot(checkpoint_path):
    return load_chatbot(checkpoint_path)

if not CHECKPOINT_PATH.exists():
    st.error(f"Checkpoint not found: {CHECKPOINT_PATH}\n\nRun the notebook's training and export cells first. Keep the artifacts folder beside this file.")
    st.stop()

try:
    model, tokenizer, device, checkpoint = cached_bot(str(CHECKPOINT_PATH))
except Exception as error:
    st.exception(error)
    st.stop()

training_config = checkpoint.get("training_config", {})
context_turns = int(training_config.get("context_turns", 3))

with st.sidebar:
    st.header("Generation settings")
    st.caption(f"Inference device: {device}")
    st.caption(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    strategy = st.radio("Decoding", ["Top-k / top-p sampling", "Greedy"], index=0)
    max_new_tokens = st.slider("Maximum response tokens", 5, 39, 25)
    temperature = st.slider("Temperature", 0.2, 1.5, 0.8, 0.05, disabled=(strategy == "Greedy"))
    top_k = st.slider("Top-k", 0, 100, 40, disabled=(strategy == "Greedy"))
    top_p = st.slider("Top-p", 0.10, 1.0, 0.90, 0.05, disabled=(strategy == "Greedy"))
    repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.12, 0.02)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("#### Try a conversation")
    st.write("Hello! · How are you? · What do you want to do today?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("latency") is not None:
            st.caption(f"Generated in {message['latency']:.2f} seconds")

prompt = st.chat_input("Write a message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    recent = [m["content"] for m in st.session_state.messages if m["role"] in ("user", "assistant")][-context_turns:]
    start = time.perf_counter()
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                answer = generate_response(
                    model, tokenizer, recent, device, checkpoint["model_config"],
                    context_turns=context_turns,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=(strategy != "Greedy"),
                    repetition_penalty=repetition_penalty,
                )
            except Exception as error:
                st.error(f"Generation failed: {error}")
                answer = "I couldn't generate a reply. Check the notebook checkpoint and logs."
        elapsed = time.perf_counter() - start
        st.markdown(answer or "[The model returned an empty response. Try a different prompt or decoding setting.]")
        st.caption(f"Generated in {elapsed:.2f} seconds")
    st.session_state.messages.append({"role": "assistant", "content": answer or "", "latency": elapsed})
