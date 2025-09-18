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
    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding]
    )

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

    st.title("📄 RAG-Powered Document Chat")
    st.write("Ask questions about the content of your uploaded PDFs.")

    # --- Sidebar Configuration ---
    st.sidebar.header("Document Selection")
    pdf_dir_path = "src"
    if not os.path.isdir(pdf_dir_path):
        st.sidebar.error(f"Directory '{pdf_dir_path}' not found. Please create it and add PDFs.")
        st.stop()
        
    pdf_files = [f for f in os.listdir(pdf_dir_path) if f.endswith(".pdf")]
    if not pdf_files:
        st.sidebar.warning("No PDF files found in the 'src' directory.")
        st.stop()
        
    selected_pdf = st.sidebar.selectbox("Select a PDF file to chat with:", pdf_files)

    ### --- KEY CHANGE: Memory Settings Removed --- ###
    # The UI and logic for memory have been removed to focus on prompt engineering.

    # Initialize session state for displaying chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input(f"Ask a question about {selected_pdf}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 1. RETRIEVE: Fetch relevant context using RAG
        with st.spinner("Analyzing document and finding relevant info..."):
            full_pdf_path = os.path.join(pdf_dir_path, selected_pdf)
            pdf_text = extract_text_from_pdf(full_pdf_path)
            
            if pdf_text:
                add_to_collection(collection, pdf_text, selected_pdf)

                openai_client = st.session_state.openai_client
                query_response = openai_client.embeddings.create(
                    input=prompt,
                    model="text-embedding-3-small"
                )
                query_embedding = query_response.data[0].embedding

                results = collection.query(query_embeddings=[query_embedding], n_results=1)
                
                if results['documents'] and results['documents'][0]:
                    retrieved_context = results['documents'][0][0]
                else:
                    retrieved_context = "No relevant content found in the document for your question."
            else:
                retrieved_context = "Could not extract text from the PDF."

        # 2. AUGMENT: Engineer the prompt with the retrieved context
        ### --- KEY CHANGE: Prompt Engineering --- ###
        # The system prompt now explicitly instructs the bot on how to behave.
        system_prompt_text = f"""
        You are an expert assistant who answers questions based on a provided document.
        Your task is to answer the user's question using ONLY the information from the 'CONTEXT FROM DOCUMENT' below.
        
        Instructions:
        1. When you use information from the context, you MUST explicitly state it. For example, start your answer with "According to the document..." or "Based on the information in '{selected_pdf}'...".
        2. If the answer is not found in the context, you MUST respond with "I'm sorry, but the answer to your question could not be found in the provided document."
        3. Do not use any of your own prior knowledge. Stick strictly to the context.
        """
        
        # The user's prompt is combined with the context for the LLM.
        final_prompt_for_api = f"""
        CONTEXT FROM DOCUMENT '{selected_pdf}':
        ---
        {retrieved_context}
        ---

        USER'S QUESTION: {prompt}
        """

        # 3. GENERATE: Send the engineered prompt to the LLM
        try:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                client = st.session_state.openai_client
                
                ### --- KEY CHANGE: Updated LLM and Simplified API Call --- ###
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Using a powerful, modern model
                    messages=[
                        {"role": "system", "content": system_prompt_text},
                        {"role": "user", "content": final_prompt_for_api}
                    ],
                    max_tokens=2048,
                    temperature=0.3 # Lower temperature for more factual, less creative answers
                )
                full_response_content = response.choices[0].message.content
                message_placeholder.markdown(full_response_content)

                st.session_state.messages.append({"role": "assistant", "content": full_response_content})

        except Exception as e:
            st.error(f"An error occurred with the AI provider: {e}")

if __name__ == "__main__":
    main()