import streamlit as st
from openai import OpenAI

def main():
    """
    Main function to run the Streamlit chatbot application.
    This version includes a conversation buffer to manage token usage.
    """
    st.title("🤖 GPT-5 Chatbot with a Memory Buffer")
    st.write("I'm a streaming chatbot that only remembers the last two turns of our conversation.")

    # --- Sidebar for API Key Configuration ---
    st.sidebar.header("Configuration")
    st.sidebar.info(
        "For this app to work, you need to set your OpenAI API key in "
        "Streamlit's secrets. Create a file at `.streamlit/secrets.toml` "
        "and add your key like this:\n\n"
        "`OPENAI_API_KEY = \"sk-...\"`"
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
    # The session_state is a Streamlit feature that preserves state across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Display Existing Chat Messages ---
    # Loop through the messages stored in the session state and display them.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle New User Input ---
    # The chat_input widget is always visible at the bottom of the screen.
    if prompt := st.chat_input("What would you like to ask?"):
        
        # 1. Add user's message to the full chat history for display.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Get the assistant's response.
        try:
            with st.spinner("Thinking..."):
                
                # --- Create the conversation buffer ---
                # We keep the full history in session_state for display purposes.
                # However, we only send the last 4 messages (2 user + 2 assistant) to the API.
                # This is the "conversation buffer".
                messages_to_send = st.session_state.messages[-4:]

                # Create the API call to OpenAI using the buffer
                stream = client.chat.completions.create(
                    model="gpt-5-chat-latest",
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in messages_to_send
                    ],
                    stream=True,
                )
                
                # Use write_stream to display the response incrementally
                with st.chat_message("assistant"):
                    response = st.write_stream(stream)

            # 3. Add the complete assistant's response to the full chat history for display.
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        except Exception as e:
            st.error(f"An error occurred while communicating with the OpenAI API: {e}")


if __name__ == "__main__":
    main()