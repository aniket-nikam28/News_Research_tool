# News_Research_tool
A Streamlit-based web application for AI-powered news article analysis. Users can input up to three news article URLs, which are processed using LangChain's RAG pipeline with Hugging Face's all-MiniLM-L6-v2 embeddings and Google's Gemini 1.5 Flash LLM. The app extracts content, creates a FAISS vector store, and answers user queries .
RockyBot: News Research Tool
RockyBot is a Streamlit-based web application that leverages LangChain's Retrieval-Augmented Generation (RAG) pipeline to analyze news articles. Users can input up to three news article URLs, which are processed to create a searchable knowledge base using Hugging Face's all-MiniLM-L6-v2 embeddings and a FAISS vector store. Queries are answered with Google's Gemini 1.5 Flash LLM, providing concise responses with cited sources. The app features a professional UI with custom styling, progress spinners, and error handling, making it ideal for journalists, researchers, and analysts.


## Features

URL Processing: Load and extract content from up to three news article URLs using UnstructuredURLLoader.
RAG Pipeline: Splits articles into chunks, generates embeddings with sentence-transformers/all-MiniLM-L6-v2, and stores them in a FAISS vector store.
AI-Powered Q&A: Answers user queries with Google's Gemini 1.5 Flash LLM, including source citations.
Professional UI: Streamlit interface with custom CSS, progress spinners, and user-friendly error messages.
Local Embeddings: Uses free, open-source Hugging Face embeddings to avoid API costs.

/Installation

Clone the Repository:
git clone https://github.com/your-username/rockybot-news-research-tool.git
cd rockybot-news-research-tool


Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install streamlit langchain langchain-google-genai langchain-huggingface sentence-transformers faiss-cpu python-dotenv unstructured


Configure Google API Key:

Sign up at console.cloud.google.com and enable the Gemini API.
Create a .env file in the project root:GOOGLE_API_KEY=your-google-api-key





Usage

Run the App:
streamlit run rag_with_gemini.py


Interact:

Open http://localhost:8501 in your browser.
Enter up to three news article URLs in the sidebar (e.g., https://www.bbc.com/news/world-12345678).
Click "Process URLs" to load and index the articles.
Ask a question (e.g., "What are the key events in these articles?") and submit to view the answer and sources.



Project Structure

rag_with_gemini.py: Main Streamlit app with the RAG pipeline.
faiss_store_huggingface.pkl: Generated FAISS index for the vector store.
.env: Stores the Google API key (not tracked in Git).

Requirements

Python 3.8+
Google Cloud account with Gemini API enabled
Minimum 4GB RAM (8GB+ recommended for large articles)

Limitations

UnstructuredURLLoader may struggle with paywalled or JavaScript-heavy sites.
Gemini API free tier has rate limits; monitor usage at console.cloud.google.com.
Local embeddings require sufficient CPU resources.

Contributing
Contributions are welcome! Please submit a pull request or open an issue for bugs, features, or improvements.
License
This project is licensed under the MIT License.
