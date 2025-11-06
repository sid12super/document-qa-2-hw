import streamlit as st
import openai
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import os

# --- 1. PDF Processing Functions ---

def process_uploaded_pdfs(uploaded_files: list, openai_api_key: str) -> FAISS:
    """
    Processes uploaded PDF files, splits them into chunks, creates embeddings,
    and stores them in a FAISS vector store.
    """
    all_chunks = []
    
    # Initialize a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        st.write(f"Processing {uploaded_file.name}...")
        # Read the PDF file
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split the extracted text into chunks
        chunks = text_splitter.split_text(text)
        
        # Create Document objects with metadata (the file name)
        for chunk in chunks:
            all_chunks.append(Document(
                page_content=chunk, 
                metadata={"source": uploaded_file.name}
            ))
    
    # Create embeddings and the vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    st.write("Processing complete!")
    return vector_store

# --- 2. RAG Conversation Handler ---

def run_rag_conversation(
    user_prompt: str, 
    vector_store: FAISS, 
    client: openai.OpenAI, 
    model_name: str, 
    max_chunks: int,
    system_prompt: str
) -> (str, list):
    """
    Runs the RAG pipeline:
    1. Retrieve relevant chunks.
    2. Augment the prompt.
    3. Generate a response from the LLM.
    """
    
    # 1. Retrieve: Get the 'k' most relevant chunks from the vector store
    # This is the "Re-Ranking" step your lab refers to.
    retrieved_chunks = vector_store.similarity_search(user_prompt, k=max_chunks)
    
    # 2. Augment: Format the chunks as context for the LLM
    context = "\n\n---\n\n".join(
        [f"Source: {chunk.metadata['source']}\n\nContent: {chunk.page_content}" for chunk in retrieved_chunks]
    )
    
    # Create the final prompt for the LLM
    augmented_prompt = f"""
    Context from documents:
    ---
    {context}
    ---
    
    User's Question: {user_prompt}
    """
    
    # 3. Generate: Call the OpenAI API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": augmented_prompt}
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0 # We want factual answers based on the context
    )
    
    return response.choices[0].message.content, retrieved_chunks

# --- 3. The Main Streamlit App ---

def main():
    st.title("SEC 10-Q RAG Chatbot")
    st.write("Upload 10-Q filings and ask questions about their financial performance.")

    # --- API Key Check ---
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        client = openai.OpenAI(api_key=openai_api_key)
    except KeyError:
        st.error("OPENAI_API_KEY not found in Streamlit secrets! Please add it.")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        st.stop()

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection (similar to lab5.py)
        selected_model = st.selectbox(
            "Choose an OpenAI Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"] # Added more common models
        )
        
        # File uploader for PDFs
        uploaded_files = st.file_uploader(
            "Upload SEC 10-Q PDF files", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        # "Process Files" button
        if st.button("Process Files") and uploaded_files:
            with st.spinner("Processing PDFs... This may take a moment."):
                try:
                    # Create and store the vector store in session state
                    st.session_state.vector_store = process_uploaded_pdfs(
                        uploaded_files, 
                        openai_api_key
                    )
                    st.success("Files processed and ready for questions!")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
        
        st.divider()

        # Re-Ranking Slider (only show if files are processed)
        if "vector_store" in st.session_state:
            max_chunks = st.slider(
                "Max chunks to retrieve (Re-Ranking)", 
                min_value=1, 
                max_value=10, 
                value=3,
                help="Controls how many relevant document chunks are used to answer your question."
            )
        else:
            max_chunks = 3 # Default value

    # --- System Prompt Definition ---
    system_prompt = (
        "You are an expert financial analyst assistant. You must answer questions based *only* "
        "on the context provided from the SEC 10-Q filings. "
        "Do not use any outside knowledge. "
        "When you provide an answer, cite the source file(s) you used. "
        "If the answer is not found in the provided context, "
        "state that 'The information was not found in the provided documents.'"
    )

    # --- Main App Logic (Q&A) ---
    
    # Only show the chat interface if the vector store has been created
    if "vector_store" in st.session_state:
        st.info("Your documents are ready. Ask a question below.")
        
        user_input = st.text_input(
            "Ask a question (e.g., 'What are the main risks for Apple?' or 'Compare Amazon's revenue to last quarter.')"
        )

        if st.button("Get Answer"):
            if not user_input:
                st.warning("Please enter a question.")
                return

            try:
                with st.spinner(f"Contacting {selected_model}..."):
                    
                    # Get the RAG response
                    answer, sources = run_rag_conversation(
                        user_prompt=user_input,
                        vector_store=st.session_state.vector_store,
                        client=client,
                        model_name=selected_model,
                        max_chunks=max_chunks,
                        system_prompt=system_prompt
                    )
                    
                    st.markdown("### 🤖 Answer")
                    st.markdown(answer)
                    
                    # Display the sources used (for Part 1 of your submission)
                    st.markdown("---")
                    st.markdown("### 📚 Sources Used")
                    with st.expander("Click to see the relevant document chunks"):
                        for chunk in sources:
                            st.info(f"**Source:** {chunk.metadata['source']}")
                            st.markdown(chunk.page_content)
                            st.markdown("---")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    
    else:
        st.warning("Please upload and process PDF files in the sidebar to begin.")

if __name__ == "__main__":
    main()