import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader

# Fix for working with ChromaDB and Streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Initialize ChromaDB client
chroma_db_path = "./ChromaDB_for_lab"
chroma_client = chromadb.PersistentClient(chroma_db_path)
collection = chroma_client.get_or_create_collection("Lab4Collection")

# Create an OpenAI client.
if 'openai_client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.openai_client = OpenAI(api_key=api_key)

def add_to_collection(collection, text, filename):
    # Create an embedding
    openai_client = st.session_state.openai_client
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    # Get the embedding
    embedding = response.data[0].embedding

    # Add embedding and document to ChromaDB
    collection.add(
        documents=[text],
        ids = [filename],
        embeddings=[embedding]
    )

def extract_text_from_pdf(file_path):
    try:
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Main application
def main():
    st.title("🤖 PDF-Based AI Chatbot")
    st.write("Ask questions about the content of the PDFs.")

    # --- Sidebar Configuration ---
    st.sidebar.header("PDF Selection")
    pdf_file_path = os.path.join(os.getcwd(), "src")
    pdf_files = [file for file in os.listdir(pdf_file_path) if file.endswith(".pdf")]
    selected_pdf = st.sidebar.selectbox("Select a PDF file", pdf_files)

    # --- Model Configuration ---
    st.sidebar.header("Model Configuration")
    llm_provider = st.sidebar.selectbox(
        "Choose LLM Provider:",
        ("OpenAI",),
    )

    # --- Memory Settings ---
    st.sidebar.header("Memory Settings")
    memory_type = st.sidebar.radio(
        "Choose conversation memory type:",
        ("Buffer of 6 messages", "Conversation Summary", "Buffer of 2,000 tokens"),
    )

    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_summary" not in st.session_state:
        st.session_state.conversation_summary = ""
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle New User Input ---
    if prompt := st.chat_input("Ask a question about the content of the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Fetch and process content from PDF ---
        with st.spinner("Fetching and processing content from PDF..."):
            pdf_file_path = os.path.join(os.getcwd(), "src", selected_pdf)
            content = extract_text_from_pdf(pdf_file_path)
            pdf_context = f"CONTENT FROM PDF:\n{content}"

        # --- Add document to ChromaDB collection ---
        add_to_collection(collection, content, selected_pdf)

        # --- Build conversation buffer ---
        history = st.session_state.messages[:-1]
        history_buffer = []

        if memory_type == "Buffer of 6 messages":
            history_buffer = history[-6:]
        elif memory_type == "Buffer of 2,000 tokens":
            current_tokens = 0
            for msg in reversed(history):
                msg_tokens = len(msg["content"].split())
                if current_tokens + msg_tokens <= 2000:
                    history_buffer.insert(0, msg)
                    current_tokens += msg_tokens
                else:
                    break

        # --- Construct final prompt based on provider and memory ---
        system_prompt_text = "You are a helpful assistant. Answer the user's question based on the provided PDF content and conversation history. If the answer is not in the content, say so."

        final_messages_for_api = []
        final_messages_for_api.append({"role": "system", "content": system_prompt_text})

        if memory_type == "Conversation Summary" and st.session_state.conversation_summary:
            final_messages_for_api.append({"role": "system", "content": f"Here is a summary of the conversation so far: {st.session_state.conversation_summary}"})
        else:
            final_messages_for_api.extend(history_buffer)

        final_messages_for_api.append({"role": "user", "content": f"CONTEXT:\n{pdf_context}\n\nQUESTION:\n{prompt}"})

        # --- API Call and Streaming Response ---
        try:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response_content = ""

                client = st.session_state.openai_client
                stream = client.chat.completions.create(model="text-davinci-002", messages=final_messages_for_api, stream=True)
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response_content += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response_content + "▌")

                message_placeholder.markdown(full_response_content)

            st.session_state.messages.append({"role": "assistant", "content": full_response_content})

            if memory_type == "Conversation Summary" and st.secrets.get("OPENAI_API_KEY"):
                with st.spinner("Creating conversation summary..."):
                    summary_prompt = "Please create a concise summary of the following conversation for your own memory."
                    conversation_for_summary = st.session_state.messages

                    client = st.session_state.openai_client
                    response = client.chat.completions.create(model="text-davinci-002", messages=[{"role": "system", "content": summary_prompt}, *conversation_for_summary])
                    st.session_state.conversation_summary = response.choices[0].message.content

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()