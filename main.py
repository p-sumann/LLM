import os

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings

from data_ingestion import load_documents
from qdrant_db import qdrant_vector_db
from vector_index import vector_index
import logging

logging.basicConfig(
                    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
                    ,level=logging.INFO,
                    filename="llm.log",
                    filemode="w")

FILE_PATH = './data/docs/'

# OPENAI Configuration
api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = api_key
llm = OpenAI()
Settings.llm = llm

# Hugging Face Embedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# set memory buffer for chat conversation history for now the token limit is 3900.
# can be increased as per need
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

def main():
    
    #ingest the documents
    logging.info('Ingesting The Data.')
    documents = load_documents(FILE_PATH)
    
    # qdrant client and vector store
    logging.info("Connecting to Qdrant Instance")
    vector_store =qdrant_vector_db()
    
    # get the vector index
    logging.info("Creating the Vector indices")
    index = vector_index(vector_store=vector_store, data_loader=documents)
    
    
    # create a chat engine that will take the prompt
    logging.info('Creating Chat Engine')
    chat_engine = index.as_chat_engine(
                    chat_mode='context',
                    memory=memory,
                    system_prompt=(
                    "You are a chatbot, able to have normal interactions with previous conversations, as well as talk"
                    " Help to generate the given query prompt \n{query}"
                    "Don't use your knowledge except for the context"),
                    verbose=True)
    
    logging.info("All OK.")
    #ask the user for Input prompt
    while True:
        query = input("\nAsk your Question: ")
        res = chat_engine.chat(query)
        history = [(query,res)]
        logging.info(history)
        print("\n")
        print(f"Response: \n\n{res}")
        print('\nThank you')
        print('*'*120,'\n')
        
if __name__ == '__main__':
    
    print('The data is pdf file of research paper "Attention is All you need".')
    main()