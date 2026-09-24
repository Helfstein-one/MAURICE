import re

import requests
import streamlit as st

API_URL = "http://localhost:8000/v1/chat/completions"

st.set_page_config(page_title="MAURICE Reasoning UI", layout="wide")

st.title("MAURICE Reasoning UI")
st.markdown("Visual interface to showcase the reasoning process of the model.")

variant = st.selectbox(
    "Select Model Variant",
    ["maurice-final", "maurice-c", "maurice-r", "maurice-g"],
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            content = msg["content"]
            # Parse <think> blocks
            parts = re.split(r"(<think>.*?</think>)", content, flags=re.DOTALL)
            for part in parts:
                if part.startswith("<think>") and part.endswith("</think>"):
                    think_content = part[7:-8].strip()
                    with st.expander("Reasoning Process"):
                        st.markdown(think_content)
                elif part.strip():
                    st.markdown(part)
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Call API
        payload = {
            "model": variant,
            "messages": st.session_state.messages,
            "stream": False,
        }
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            assistant_content = result["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            assistant_content = f"Error connecting to inference server: {e}"

        # Render assistant content
        parts = re.split(r"(<think>.*?</think>)", assistant_content, flags=re.DOTALL)
        for part in parts:
            if part.startswith("<think>") and part.endswith("</think>"):
                think_content = part[7:-8].strip()
                with st.expander("Reasoning Process"):
                    st.markdown(think_content)
            elif part.strip():
                st.markdown(part)

        st.session_state.messages.append({"role": "assistant", "content": assistant_content})
