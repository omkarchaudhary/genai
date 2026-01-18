import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

#Streamlit app title
st.set_page_config(
    page_title="Smart Chatbot", 
    page_icon="🤖",
    layout="centered",
    )
st.title("🤖 Smart Chatbot")

#initiate chat history and check if it exists in session state
if "chat_history" not in st.session_state:
     st.session_state.chat_history = []


#Show chat history

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# initialize Groq chat model
llm = ChatGroq(model = "llama-3.3-70b-versatile",
               temperature=0.0,
                groq_api_key=os.getenv("GROQ_API_KEY")
               )

user_prompt =  st.chat_input("Ask me anything...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    response = llm.invoke(
        input=[
            {"role": "system", "content": "You are a helpful assistant"},
            *st.session_state.chat_history,
        ]
    )

    assistance_response = response.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistance_response})

    with st.chat_message("assistant"):
        st.markdown(assistance_response)