import streamlit as st
import PyPDF2
from openai import OpenAI

# Show title and description.
st.title("📄 Sid's Document Question Answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
)

# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Add a model selector.
    model = st.selectbox(
        "Select a GPT model",
        options=["gpt-3.5-turbo", "gpt-4.1", "gpt-5-nano", "gpt-5-chat-latest"],
        index=0,  # Default to "gpt-3.5-turbo"
    )

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # Process the uploaded file.
        file_extension = uploaded_file.name.split('.')[-1]
        document = None
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                document = "".join(page.extract_text() for page in pdf_reader.pages)
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")

        if document:
            # Prepare the OpenAI API request.
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            # Generate an answer using the OpenAI API with streaming.
            try:
                response_stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )

                # Stream the response to the app.
                st.write("### Response:")
                response_text = ""
                for chunk in response_stream:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            response_text += delta["content"]
                            st.write(delta["content"], end="")

                if not response_text:
                    st.error("No response received from the model.")
            except Exception as e:
                st.error(f"Error generating response: {e}")


