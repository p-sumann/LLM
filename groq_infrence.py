from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate


load_dotenv()

laoder = WebBaseLoader('https://bitskraft.com/career/')
docs = laoder.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)

documents = splitter.split_documents(docs)

ollama_emb = OllamaEmbeddings(
    model="llama:7b",
)

vector_db = FAISS.from_documents(documents, embedding=ollama_emb)


llm = chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

