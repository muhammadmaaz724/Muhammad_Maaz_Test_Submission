import streamlit as st
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

def ensure_event_loop():
    """
    Ensures an asyncio event loop exists (useful for Streamlit and threaded environments).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

@st.cache_resource
def load_retriever():
    """
    Loads the FAISS vector store and returns a retriever for semantic search.

    Returns:
        retriever: A retriever object to search the embedded transcript chunks.
    """
    ensure_event_loop()

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = FAISS.load_local(
        "vectorstore/faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def chat_with_transcript(question: str):
    """
    Retrieves relevant chunks from the transcript and generates a response using the LLM.

    Args:
        question (str): The user's input question.

    Returns:
        str: LLM-generated answer based on the retrieved transcript context.
    """
    retriever = load_retriever()
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return "Sorry, I couldn't find anything relevant in the transcript."

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs).strip()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful assistant analyzing customer support transcripts. Use the provided transcript context to answer questions accurately and concisely.")
        ]
    
    chat_history = st.session_state.chat_history

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Based on the following transcript context, please answer the question:\n\nTranscript Context:\n{context}\n\nQuestion: {question}")
    ])

    chain = prompt | llm

    response = chain.invoke({
        "chat_history": chat_history,
        "context": context_text,
        "question": question
    })

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=response.content))

    return response.content
