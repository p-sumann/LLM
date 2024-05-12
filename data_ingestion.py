from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

import os

def load_documents(file_path: str):
    """ Function to load the document from file path

    Args:
        file_path (str): path of file

    Returns:
        loaded_data: returns the loaded documents
    """
    documents = SimpleDirectoryReader(file_path)
    loaded_data = documents.load_data()
    
    return loaded_data

# def parser(documents: list):
#     """ Function that parses the documents

#     Args:
#         documents (list): input documents like pdf, csv, xml etc

#     Returns:
#         nodes: returns the parsed version of documents into chunks
#     """
#     # splits the documents in chunks (size 1024 and overlap 20)
#     node_parser = SentenceSplitter(chunk_size=1024,chunk_overlap=20)
#     # parse the documents into nodes
#     nodes = node_parser.get_nodes_from_documents(documents=documents, show_progress=False)   
#     print(f"Spliting data into chunks. Chunk Size is: {len(nodes)}")
#     return nodes