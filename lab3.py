import streamlit as st
from openai import OpenAI
import tiktoken

def get_token_count(text, model="gpt-5-chat-latest"):
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def main():
    """
    Main function to run the Streamlit chatbot application.
    This version includes a selectable conversation buffer to manage token usage.
    """
    st.title("🤖 GPT-5 Chatbot with Memory Buffer")
    st.write("I'm a streaming chatbot with a configurable memory buffer.")

    # --- Sidebar for API Key and Buffer Configuration ---
    st.sidebar.header("Configuration")
    st.sidebar.info(
        "For this app to work, you need to set your OpenAI API key in "
        "Streamlit's secrets. Create a file at `.streamlit/secrets.toml` "
        "and add your key like this:\n\n"
        "`OPENAI_API_KEY = \"sk-...\"`"
    )

    st.sidebar.header("Buffer Settings")
    buffer_type = st.sidebar.radio(
        "Choose buffer type:",
        ("Message Count (Default)", "Token Limit")
    )

    max_tokens = 0
    if buffer_type == "Token Limit":
        max_tokens = st.sidebar.number_input(
            "Max tokens for buffer:",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100
        )

    # --- Retrieve API Key and Initialize Client ---
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please follow the instructions in the sidebar.")
            st.stop()
        
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        st.stop()

    # --- Initialize Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Display Existing Chat Messages ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle New User Input ---
    if prompt := st.chat_input("What would you like to ask?"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("Thinking..."):
                
                # --- Create the conversation buffer based on sidebar selection ---
                messages_to_send = []
                if buffer_type == "Token Limit":
                    current_tokens = 0
                    # Iterate through messages in reverse to get the most recent ones
                    for msg in reversed(st.session_state.messages):
                        msg_tokens = get_token_count(msg["content"])
                        if current_tokens + msg_tokens <= max_tokens:
                            messages_to_send.insert(0, msg) # Insert at the beginning
                            current_tokens += msg_tokens
                        else:
                            break # Stop when we exceed the token limit
                else: # Default to Message Count buffer
                    messages_to_send = st.session_state.messages[-4:]

                # Create the API call to OpenAI using the selected buffer
                stream = client.chat.completions.create(
                    model="gpt-5-chat-latest",
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in messages_to_send
                    ],
                    stream=True,
                )
                
                with st.chat_message("assistant"):
                    response = st.write_stream(stream)

            st.session_state.messages.append({"role": "assistant", "content": response})
        
        except Exception as e:
            st.error(f"An error occurred while communicating with the OpenAI API: {e}")


if __name__ == "__main__":
    main()