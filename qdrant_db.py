import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore


def qdrant_vector_db():
    """ 
    function for connecting and creating qdrant vector db

    Returns:
        vector_store: returns vectore store 
    """
    # intialize the qdrant client on local sytem.
    # can use on cloud or memory or some path as well. 
    # for this demo, I have used on local path called 'qdrant' where the qdrant db will setup.
    client = qdrant_client.QdrantClient(path="qdrant")
    
    # vector store to store vector emebeddings
    vector_store = QdrantVectorStore(
                    client=client,
                    collection_name='frost'
                    )
    
    
    return vector_store


