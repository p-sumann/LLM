import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.environ['GROQ_API_KEY']

load_dotenv()

laoder = WebBaseLoader('https://bitskraft.com/career/')
docs = laoder.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

documents = splitter.split_documents(docs)

embeddings = OllamaEmbeddings()

vector_db = FAISS.from_documents(documents, embedding=embeddings)


llm = chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=groq_api_key)

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


retrieval_chain.invoke({"input":prompt})
