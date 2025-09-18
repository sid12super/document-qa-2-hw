import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
import sys

# Fix for working with ChromaDB and Streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# --- Global ChromaDB Initialization ---
# The client is persistent, so it's efficient to initialize it once globally.
chroma_db_path = "./ChromaDB_for_lab"
chroma_client = chromadb.PersistentClient(chroma_db_path)
collection = chroma_client.get_or_create_collection("Lab4Collection")

def add_to_collection(collection, text, filename):
    """Adds a document and its embedding to the ChromaDB collection."""
    openai_client = st.session_state.openai_client
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(documents=[text], ids=[filename], embeddings=[embedding])

def extract_text_from_pdf(file_path):
    """Extracts all text from a given PDF file."""
    try:
        pdf_reader = PdfReader(file_path)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

### --- KEY CHANGE: New function for efficient, one-time processing --- ###
def setup_document_collection(collection, pdf_file_path, pdf_filename):
    """
    Clears the collection and ingests a new document.
    This ensures we only process and embed each document once per selection.
    """
    with st.spinner(f"Processing and embedding '{pdf_filename}'... This happens only once per document."):
        # 1. Clear any old documents from the collection
        count = collection.count()
        if count > 0:
            ids_to_delete = collection.get(limit=count)['ids']
            collection.delete(ids=ids_to_delete)

        # 2. Extract text and add the new document
        text = extract_text_from_pdf(pdf_file_path)
        if text:
            add_to_collection(collection, text, pdf_filename)
            st.session_state.processed_pdf = pdf_filename # Mark this PDF as processed
            st.success(f"'{pdf_filename}' is now ready for questions.", icon="✅")
        else:
            st.error("Failed to process the document.")


# --- Main Application Logic ---
def main():
    if 'openai_client' not in st.session_state:
        try:
            st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client. Check API key. Error: {e}")
            st.stop()

    st.title("📄 AI Chat Application")
    st.write("Switch between chatting with your documents (RAG) or having a general conversation.")

    st.sidebar.header("Settings")
    chat_mode = st.sidebar.radio(
        "Choose your chat mode:",
        ("Document Q&A (RAG)", "General Chat")
    )

    if chat_mode == "Document Q&A (RAG)":
        st.sidebar.subheader("Document Selection")
        pdf_dir_path = "src"
        if not os.path.isdir(pdf_dir_path):
            st.sidebar.error(f"Directory '{pdf_dir_path}' not found.")
            st.stop()
        
        pdf_files = [f for f in os.listdir(pdf_dir_path) if f.endswith(".pdf")]
        if not pdf_files:
            st.sidebar.warning("No PDF files found in 'src' directory.")
            st.stop()
            
        selected_pdf = st.sidebar.selectbox("Select a PDF file to chat with:", pdf_files)

        ### --- KEY CHANGE: Check if the document needs to be processed --- ###
        # This logic uses session_state to avoid re-embedding the same file.
        if "processed_pdf" not in st.session_state or st.session_state.processed_pdf != selected_pdf:
            full_pdf_path = os.path.join(pdf_dir_path, selected_pdf)
            setup_document_collection(collection, full_pdf_path, selected_pdf)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Mode 1: Document Q&A (RAG) ---
        if chat_mode == "Document Q&A (RAG)":
            with st.spinner("Searching for relevant information..."):
                # The document is already embedded, so we just query it.
                openai_client = st.session_state.openai_client
                query_response = openai_client.embeddings.create(input=prompt, model="text-embedding-3-small")
                query_embedding = query_response.data[0].embedding
                results = collection.query(query_embeddings=[query_embedding], n_results=1)
                
                retrieved_context = results['documents'][0][0] if results.get('documents') and results['documents'][0] else "No relevant content found."

            system_prompt_text = "..." # Remainder of logic is the same

        # --- Mode 2: General Chat ---
        else:
            # General chat logic remains the same
            system_prompt_text = "You are a helpful AI assistant."
            messages_for_api = [{"role": "system", "content": system_prompt_text}] + st.session_state.messages
        
        # (The unified API call logic remains the same as before)
        try:
            client = st.session_state.openai_client
            # ... API call ...
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()