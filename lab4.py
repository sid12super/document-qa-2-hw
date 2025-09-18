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

# --- Main Application Logic ---
def main():
    # Initialize OpenAI client once per session
    if 'openai_client' not in st.session_state:
        try:
            st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client. Check your API key. Error: {e}")
            st.stop()

    st.title("📄 AI Chat Application")
    st.write("Switch between chatting with your documents (RAG) or having a general conversation.")

    # --- Sidebar Configuration ---
    st.sidebar.header("Settings")
    
    ### --- KEY CHANGE: Chat Mode Selector --- ###
    chat_mode = st.sidebar.radio(
        "Choose your chat mode:",
        ("Document Q&A (RAG)", "General Chat")
    )

    # Document selection is only needed for RAG mode
    if chat_mode == "Document Q&A (RAG)":
        st.sidebar.subheader("Document Selection")
        pdf_dir_path = "src"
        if not os.path.isdir(pdf_dir_path):
            st.sidebar.error(f"Directory '{pdf_dir_path}' not found.")
            st.stop()
        
        pdf_files = [f for f in os.listdir(pdf_dir_path) if f.endswith(".pdf")]
        if not pdf_files:
            st.sidebar.warning("No PDF files found in the 'src' directory.")
            st.stop()
            
        selected_pdf = st.sidebar.selectbox("Select a PDF file to chat with:", pdf_files)
    else:
        selected_pdf = None

    # Initialize session state for displaying chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle New User Input based on Chat Mode ---
    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Mode 1: Document Q&A (RAG) ---
        if chat_mode == "Document Q&A (RAG)":
            with st.spinner("Analyzing document and finding relevant info..."):
                full_pdf_path = os.path.join(pdf_dir_path, selected_pdf)
                pdf_text = extract_text_from_pdf(full_pdf_path)
                
                if pdf_text:
                    add_to_collection(collection, pdf_text, selected_pdf)
                    
                    openai_client = st.session_state.openai_client
                    query_response = openai_client.embeddings.create(
                        input=prompt, model="text-embedding-3-small"
                    )
                    query_embedding = query_response.data[0].embedding
                    results = collection.query(query_embeddings=[query_embedding], n_results=1)
                    
                    retrieved_context = results['documents'][0][0] if results['documents'] and results['documents'][0] else "No relevant content found."
                else:
                    retrieved_context = "Could not extract text from the PDF."

            system_prompt_text = f"""
            You are an expert assistant. Answer the user's question using ONLY the context below.
            - If you use the context, start with "According to the document...".
            - If the answer isn't in the context, say so clearly.
            - Do not use any prior knowledge.
            """
            
            final_prompt_for_api = f"CONTEXT FROM '{selected_pdf}':\n{retrieved_context}\n\nUSER'S QUESTION: {prompt}"
            
            messages_for_api = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": final_prompt_for_api}
            ]

        # --- Mode 2: General Chat ---
        else:
            with st.spinner("Thinking..."):
                system_prompt_text = "You are a helpful and friendly AI assistant. Answer the user's questions conversationally."
                # For general chat, we include the conversation history
                messages_for_api = [{"role": "system", "content": system_prompt_text}] + st.session_state.messages

        # --- Unified API Call and Response Handling ---
        try:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                client = st.session_state.openai_client
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages_for_api,
                    max_tokens=2048,
                    temperature=0.7 
                )
                full_response_content = response.choices[0].message.content
                message_placeholder.markdown(full_response_content)
                
                # Add assistant's response to the end of the message list
                st.session_state.messages.append({"role": "assistant", "content": full_response_content})

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()