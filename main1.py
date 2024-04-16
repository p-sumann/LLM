from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage import StorageContext

import os

api_key = os.getenv('OPENAI_API_KEY')

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# llm = Gemini(api_key=api_key, model_name='models/gemini-pro')

os.environ['OPENAI_API_KEY'] = api_key
llm = OpenAI()
Settings.llm = llm
FILE_PATH = './data/docs/'

def main():
    docs = SimpleDirectoryReader(FILE_PATH)
    data_loader = docs.load_data()

    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = node_parser.get_nodes_from_documents(
        documents=data_loader, show_progress=False
    )
    print(f"Spliting data into chunks. Chunk Size is: {len(nodes)}")
    # vector database
    client = qdrant_client.QdrantClient(
        path="qdrant",
    )
    
    # vector store to store vector
    vector_store = QdrantVectorStore(
        client=client,
        collection_name='frost', 
    )
    

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
                        documents=data_loader,
                        transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)],
                        storage_context=storage_context)
    # index = VectorStoreIndex.from_vector_store(
    #                     vector_store,
    #                     storage_context=storage_context
    #                 )
    index.storage_context.persist('./vector_index')
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    response = index.as_chat_engine(
                        chat_mode='context',
                        memory=memory,
                        system_prompt=(
                                "You are a chatbot, able to have normal interactions with previous conversations, as well as talk"
                                " Help to generate the given query prompt \n{query}"
                                ),
                        verbose=True)
    
    while True:
        query = input("Ask your Question: ")
        res = response.chat(query)
        print("\n",res)
        print('\nThank you')
        print('*'*105,'\n')
if __name__ == '__main__':
    print('Ingesting The Data.')
    print('The data is pdf file of research paper "Attention is All you need".')
    main()