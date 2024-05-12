from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", human)])

chat = ChatGroq(temperature=0, model_name="llama2-70b-4096")
prompt = ChatPromptTemplate.from_messages(
    [("human", "Write a lengthy about {topic}")])
chain = prompt | chat


import streamlit as st

for chunk in chain.stream({"topic": "Write a python demo book"}):
    print(chunk.content, end="", flush=True)
    st.write(chunk.content)
