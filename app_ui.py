import streamlit as st
from predict_news import fake_no_fake
from PIL import Image

im = Image.open('news_img.png')
st.set_page_config(page_title="Fake News Detection App", page_icon=im)
st.title("Fake News Detection App")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter the news you want to classify"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner('Detecting the article...'):
        response = f"Echo: The given news article seems to be {fake_no_fake(prompt)}."
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})