import streamlit as st      # Importing Streamlit for UI
import os
import requests     # To call external API/Libraries
from bs4 import BeautifulSoup       # Importing BeautifulSoup for Web Scraping
from langchain.text_splitter import CharacterTextSplitter       # To split the text into chunks
from langchain_community.embeddings import OllamaEmbeddings       # To get the embeddings of the text
from langchain_community.chat_models import ChatOllama       # To get the response from the chatbot
from langchain_community.vectorstores import FAISS       # To store the vectors (Vector DB)
from langchain_community.document_loaders import BSHTMLLoader       # To load the HTML document
from langchain.memory import ConversationBufferMemory      # To store the conversation history
from langchain.chains import RetrievalQA      # To get the response from the chatbot
from langchain.docstore.document import Document      # To store the document
from langchain.prompts import PromptTemplate      # To create the prompt
import numpy as np
import time
import tempfile

#Configuration variables
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MODEL_NAME = "deepseek-r1:1.5b"
TEMPERATURE = 0.4

# Intitalize session state variables
if 'qa' not in st.session_state:    # If question-answer chain DNE then set it to empty state
    st.session_state.qa = None
if 'vectorstore' not in st.session_state:   # If vectorstore DNE then set it to empty state
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:  # If chat history DNE then set it to empty state
    st.session_state.chat_history = []

def fetch_and_process_website(url):
    """Fetches and processes website content"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    # extracting text from website
    try:
        with st.spinner('Feteching website content/data...'):
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # use a temporary file to store the HTML content
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as temp_file:
                temp_file.write(response.text)
                temp_file_path = temp_file.name

            try:
                # load the HTML content
                loader = BSHTMLLoader(temp_file_path)
                documents = loader.load()
            except ImportError:
                st.warning("'lxml' is not installed. Falling back to built-in 'html.parser'.")

                # If 'lxml' is not available, use the built-in 'html.parser'
                loader = BSHTMLLoader(temp_file_path, bs_kwargs={'features': 'html.parser'})
                documents = loader.load()
            
            #cleanup the temporary file
            os.unlink(temp_file_path)

            text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            texts = text_splitter.split_documents(documents)

            return texts
        
    except Exception as e:
        st.error(f"Error processing webiste: {str(e)}")
        return None
    
# Initialize the RAG pipeline with llm
def initialize_rag_pipeline(texts):
    """Initialize the Rag pipeline with the given texts"""
    # setup ollama language model
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=TEMPERATURE
    )

    # create embeddings
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    # create vectorstore
    vectorstore = FAISS.from_documents(texts, embeddings)

    # setup the retrieval based QA system
    template = """context: {context}       

    Question: {question}

    Answer the question concisely based only on the given context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

    But, if the question is generic, then go ahead and answer the question, example what is an electric vehicle?
    """
    PROMPT = PromptTemplate(        # Create the prompt
        template=template,
        input_variables=["context", "question"]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)      # Create the memory for the conversation history

    # creating the entire chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa, vectorstore

# defining the main app
def main():
    st.title("🤖 RAG Website Query System")
    st.write("Enter a website URL to analyze the content and ask question about it's content.")

    # URL input
    url = st.text_input("Enter the website URL")

    # Process button
    if st.button("Process Website") and url:
        texts = fetch_and_process_website(url)
        if texts:
            st.success(f"Successfully processed {len(texts)} text chunks from the website.")
            st.session_state.qa, st.session_state.vectorstore = initialize_rag_pipeline(texts)
            st.session_state.chat_history = []   # Reset chat history for new website.
    
    # Show query interface only if pipeline is initialized
    if st.session_state.qa and st.session_state.vectorstore:
        st.write("___")
        st.subheader("Ask questions")

        # Query Input
        query = st.text_input("Enter your question:")

        if st.button("Ask"):
            if query:
                with st.spinner('Searching for answer...'):
                    # Get relevant documents
                    relevant_docs = st.session_state.vectorstore.similarity_search_with_score(query, k=3)

                    # Display relevant chunks in expander
                    with st.expander("View relevant chunks"):
                        for i, (doc, store) in enumerate(relevant_docs, 1):
                            st.write(f"Chunk {i} (Score: {store:.4f})")
                            st.write(doc.page_content)
                            st.write("__")
                    
                    # Get response 
                    response = st.session_state.qa.invoke({"query": query})

                    # Add to chat history
                    st.session_state.chat_history.append({"question": query, "answer": response['result']})

                    # Display chat history
                    st.write("___")
                    st.subheader("Chat History")
                    for chat in st.session_state.chat_history:
                        st.write(f"**Q:** {chat['question']}")
                        st.write(f"**A:** {chat['answer']}")
                        st.write("___")
        
    # Add sidebar with information
    with st.sidebar:
        st.subheader("ℹ️ About")
        st.write("""
            This is a RAG system that allows you to:
            1. Input any website URL
            2. Analyze the content of the website
            3. Ask questions about the content
                    
            The system uses :
            - Ollama (deepseek-r1) for text generation
            - FAISS for vector storage
            - Langchain for the RAG pipeline
            """)
        
        st.subheader("Model Configurations")
        st.write(f"**Model:** {MODEL_NAME}")
        st.write(f"**Temperature:** {TEMPERATURE}")
        st.write(f"**Chunk Size:** {CHUNK_SIZE}")
        st.write(f"**Chunk Overlap:** {CHUNK_OVERLAP}")

if __name__ == "__main__":
    main()