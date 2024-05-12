from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core.node_parser import SentenceSplitter

def vector_index(vector_store, data_loader):
    """Function to computer vector indices

    Args:
        vector_store (QdrantVectorStore): vectore database that stores the vector emebeddings
        data_loader (_type_): original documents

    Returns:
        index: returns the vector index
    """
    # contianer to store vectors
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # tranformation is done to convert the entire documents into a chunk
    index = VectorStoreIndex.from_documents(
                        documents=data_loader,
                        transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)],
                        storage_context=storage_context)
    # persit the storage context to local system
    index.storage_context.persist('./vector_index')
    
    return index
