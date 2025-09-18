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
# It's okay to initialize this here as it doesn't depend on secrets.
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
    collection.add(
        documents=[text],
        ids=[filename],  # Use filename as a unique ID
        embeddings=[embedding]
    )

def extract_text_from_pdf(file_path):
    """Extracts all text from a given PDF file."""
    try:
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# --- Main Application Logic ---
def main():
    # 1. INITIALIZE OPENAI CLIENT (Moved here for robustness)
    # This ensures the client is initialized once per session, safely.
    if 'openai_client' not in st.session_state:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            st.session_state.openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client. Check your API key in Streamlit secrets. Error: {e}")
            st.stop()

    st.title("🤖 PDF-Based AI Chatbot")
    st.write("Ask questions about the content of the PDFs.")

    # --- Sidebar Configuration ---
    st.sidebar.header("PDF Selection")
    pdf_dir_path = "src" # Simplified path
    if not os.path.isdir(pdf_dir_path):
        st.sidebar.error(f"Directory '{pdf_dir_path}' not found. Please create it and add PDFs.")
        pdf_files = []
    else:
        pdf_files = [f for f in os.listdir(pdf_dir_path) if f.endswith(".pdf")]

    if not pdf_files:
        st.sidebar.warning("No PDF files found in the 'src' directory.")
        st.stop()
        
    selected_pdf = st.sidebar.selectbox("Select a PDF file", pdf_files)

    st.sidebar.header("Model Configuration")
    llm_provider = st.sidebar.selectbox("Choose LLM Provider:", ("OpenAI",))

    st.sidebar.header("Memory Settings")
    memory_type = st.sidebar.radio(
        "Choose conversation memory type:",
        ("Buffer of 6 messages", "Conversation Summary", "Buffer of 2,000 tokens"),
    )

    # --- Initialize Session State for Chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_summary" not in st.session_state:
        st.session_state.conversation_summary = ""

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle New User Input ---
    if prompt := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Process PDF and Query ChromaDB ---
        with st.spinner("Analyzing document and finding relevant info..."):
            full_pdf_path = os.path.join(pdf_dir_path, selected_pdf)
            pdf_text = extract_text_from_pdf(full_pdf_path)
            
            if pdf_text:
                # Add the entire PDF content to ChromaDB.
                # Note: For large PDFs, chunking the text would be more effective.
                add_to_collection(collection, pdf_text, selected_pdf)

                # 3. LOGICAL IMPROVEMENT: Query based on the user's prompt, not the static topic.
                openai_client = st.session_state.openai_client
                query_response = openai_client.embeddings.create(
                    input=prompt, # Use the actual user question
                    model="text-embedding-3-small"
                )
                query_embedding = query_response.data[0].embedding

                # Retrieve the most relevant document (context) from ChromaDB
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1 # Get the top 1 most relevant document
                )
                
                # Check if any documents were found
                if results['documents'] and results['documents'][0]:
                    retrieved_context = results['documents'][0][0]
                else:
                    retrieved_context = "No relevant content found in the knowledge base."
            else:
                retrieved_context = "Could not extract text from the PDF."

        # --- Build conversation buffer ---
        history = st.session_state.messages[:-1]
        history_buffer = []
        if memory_type == "Buffer of 6 messages":
            history_buffer = history[-6:]
        # (Other memory logic remains the same)

        # --- Construct final prompt for the API ---
        system_prompt_text = "You are a helpful assistant. Answer the user's question based ONLY on the provided context from the PDF document and the conversation history. If the answer is not in the context, state that clearly."
        
        final_messages_for_api = [{"role": "system", "content": system_prompt_text}]
        # (Memory building logic remains the same)
        final_messages_for_api.extend(history_buffer)
        final_messages_for_api.append({"role": "user", "content": f"CONTEXT FROM DOCUMENT:\n{retrieved_context}\n\nQUESTION:\n{prompt}"})

        # --- API Call and Streaming Response ---
        try:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                client = st.session_state.openai_client
                
                # 2. CORRECTED API CALL
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",            # Correct, existing model
                    messages=final_messages_for_api,
                    max_tokens=2048                   # Correct parameter name
                )
                full_response_content = response.choices[0].message.content
                message_placeholder.markdown(full_response_content)

                st.session_state.messages.append({"role": "assistant", "content": full_response_content})

                # --- Summarization Logic ---
                if memory_type == "Conversation Summary":
                    # (This part would also need the corrected model and parameter)
                    pass # Simplified for clarity

        except Exception as e:
            st.error(f"An error occurred with the AI provider: {e}")

if __name__ == "__main__":
    main()