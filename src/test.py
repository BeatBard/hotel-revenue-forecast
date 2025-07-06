"""This is a chatbot application for interacting with a Sinhala LLM model."""

import streamlit as st
import requests  # type: ignore
import time

# API_URL = "http://localhost:8000/generate"  # Change if deployed elsewhere
API_URL = "https://i5ngzahxm8danp-8000.proxy.runpod.net/generate"  # Change if deployed elsewhere

st.title("Sinhala LLM Chatbot")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Prompt input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Send to FastAPI backend
            response = requests.post(API_URL, json={"question": prompt})
            response.raise_for_status()
            full_response = response.json()["response"]
        except Exception as e:
            full_response = f"❌ Error: {e}"

        # Stream response typing
        for i in range(len(full_response)):
            message_placeholder.markdown(full_response[: i + 1] + "▌")
            time.sleep(0.01)
        message_placeholder.markdown(full_response)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})