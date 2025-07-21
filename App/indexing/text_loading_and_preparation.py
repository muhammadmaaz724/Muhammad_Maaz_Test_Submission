from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio


def document_loading(path_to_input_document):
    """
    Loads a text file as a LangChain document.

    Args:
        path_to_input_document (str): Path to the input document.

    Returns:
        list: Full loaded document.
    """
    
    loader = TextLoader(path_to_input_document,encoding='utf-8')
    full_document = loader.load()
    
    return full_document

def document_splitting(path_to_input_document):
    """
    Splits a loaded document into smaller chunks to fit within LLM context limits.

    Args:
        path_to_input_document (str): Path to the input document.

    Returns:
        list: Chunks of the document.
    """
    
    full_document = document_loading(path_to_input_document)
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=100)
    chunks_of_document = splitter.split_documents(full_document)
    
    return chunks_of_document
    


def creating_and_storing_embeddings(path_to_input_document):
    """
    Creates embeddings for document chunks and stores them locally using FAISS.

    Args:
        path_to_input_document (str): Path to the input document.

    Returns:
        FAISS: Vector store containing the document embeddings.
    """
   
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    load_dotenv()
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    chunks = document_splitting(path_to_input_document)
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    vector_store.save_local("vectorstore/faiss_index")

    return vector_store
