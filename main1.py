import os
import streamlit as st
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextInput>div>input {
        border: 2px solid #007bff;
        border-radius: 5px;
        padding: 10px;
    }
    .stTextInput>label {
        font-size: 16px;
        font-weight: bold;
        color: #333;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1a1a1a;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 10px;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
        font-size: 16px;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Set page configuration for branding
st.set_page_config(page_title="News Research Tool", page_icon="📈", layout="wide")

# Title and header
st.title("News Research Tool")
st.markdown("Effortlessly research news articles with AI-powered insights.", unsafe_allow_html=True)

# Sidebar for URL inputs
with st.sidebar:
    st.header("Input News Article URLs")
    urls = []
    for i in range(3):
        url = st.text_input(f"URL {i+1}", placeholder="Enter a valid news article URL", key=f"url_{i}")
        urls.append(url)
    
    process_url_clicked = st.button("Process URLs", help="Click to load and process the URLs")

file_path = "faiss_store_huggingface.pkl"

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.9,
        max_output_tokens=500,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
except KeyError:
    st.error("Google API Key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

# Main content area
with st.container():
    st.markdown("---")
    main_placeholder = st.empty()

    if process_url_clicked:
        # Validate URLs
        if not any(urls):
            main_placeholder.error("Please provide at least one valid URL.")
        else:
            try:
                with st.spinner("Loading data..."):
                    loader = UnstructuredURLLoader(urls=[url for url in urls if url])
                    data = loader.load()
                
                with st.spinner("Splitting text..."):
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=['\n\n', '\n', '.', ','],
                        chunk_size=1000
                    )
                    docs = text_splitter.split_documents(data)
                
                with st.spinner("Building embeddings..."):
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectorstore_huggingface = FAISS.from_documents(docs, embeddings)
                
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_huggingface, f)
                
                st.markdown('<p class="success-message">URLs processed successfully!</p>', unsafe_allow_html=True)
            
            except Exception as e:
                main_placeholder.error(f"Error processing URLs: {str(e)}")
                st.stop()

    # Query input and results
    with st.form(key="query_form"):
        query = st.text_input("Ask a question about the articles:", placeholder="e.g., What are the key events in these articles?")
        submit_query = st.form_submit_button("Submit Question")
    
    if submit_query and query:
        if os.path.exists(file_path):
            try:
                with st.spinner("Generating answer..."):
                    with open(file_path, "rb") as f:
                        vectorstore = pickle.load(f)
                        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                        result = chain({"question": query}, return_only_outputs=True)
                
                st.header("Answer")
                st.write(result["answer"])
                
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.markdown(f"- {source}")
            except Exception as e:
                st.error(f"Error answering query: {str(e)}")
        else:
            st.error("No processed data found. Please process URLs first.")