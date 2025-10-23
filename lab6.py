# lab6.py
# AI Fact-Checker + Citation Builder
# This script implements Lab 6, structured with a main() function.

import streamlit as st
from openai import OpenAI
import json

# --- 1. Initialization (Global) ---
# These are defined globally so they are not re-created on every re-run.

# Initialize the OpenAI client
# (Assumes OPENAI_API_KEY is set in secrets.toml or environment)
try:
    client = OpenAI()
except Exception as e:
    # We'll show the error inside the main app function
    CLIENT_ERROR = e
else:
    CLIENT_ERROR = None

# Define the required JSON schema for the output
# This ensures we get structured, reliable data from the API
FACT_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "claim": {
            "type": "string",
            "description": "The original claim being verified."
        },
        "verdict": {
            "type": "string",
            "enum": ["True", "False", "Partly True", "Unverified"],
            "description": "The factual verdict for the claim."
        },
        "explanation": {
            "type": "string",
            "description": "A concise explanation for the verdict, based on the sources."
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string", "format": "uri"}
                },
                "required": ["title", "url"]
            },
            "description": "A list of credible sources used for verification."
        }
    },
    "required": ["claim", "verdict", "explanation", "sources"]
}

# Initialize session state for history (Lab 6d enhancement)
if 'claim_history' not in st.session_state:
    st.session_state.claim_history = []

# --- 2. Core Fact-Checker Function (Lab 6b) ---

def fact_check_claim(user_claim: str):
    """
    Calls the OpenAI Responses API to fact-check a claim using web_search
    and returns a guaranteed JSON object.
    """
    
    # System prompt as defined in the lab instructions
    system_prompt = """
    You are a factual verification assistant.
    For any given claim, search the web for credible sources and return
    a JSON object that adheres to the provided JSON schema.
    
    The JSON object must contain:
    - claim: The original claim
    - verdict: True / False / Partly True / Unverified
    - explanation: Your reasoning
    - sources: A list of {title, url} objects
    """
    
    try:
        # Use client.responses.create
        response = client.responses.create(
            model="gpt-4.1", # As specified in the lab instructions
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_claim}
            ],
            # Use the web_search tool
            tools=[{"type": "web_search"}],
            # Enforce the JSON schema
            response_format={
                "type": "json_schema",
                "json_schema": FACT_CHECK_SCHEMA
            }
        )
        
        # The API guarantees the output is a valid JSON string
        return response.output_text
        
    except Exception as e:
        st.error(f"An error occurred while calling the API: {e}")
        return None

# --- 3. Streamlit UI (Main Function) ---

def main():
    """
    Main function to run the Streamlit UI for Lab 6.
    """
    
    # Check if client initialized correctly
    if CLIENT_ERROR:
        st.error(f"Failed to initialize OpenAI client. Ensure your API key is set.\n{CLIENT_ERROR}")
        st.stop()

    # Page Title (Lab 6a)
    st.title("🤖 AI Fact-Checker + Citation Builder")
    st.markdown("---")

    # Input section (Lab 6a)
    user_claim = st.text_input("Enter a factual claim to verify:", 
                               placeholder="e.g., Is dark chocolate actually healthy?", 
                               key="lab6_user_claim")

    # Button (Lab 6a)
    if st.button("Check Fact", key="lab6_check_fact"):
        if user_claim:
            # Show spinner while working (Lab 6c)
            with st.spinner("Verifying... Searching sources and reasoning..."):
                # Call the fact-check function (Lab 6c)
                result_json_string = fact_check_claim(user_claim)
                
                if result_json_string:
                    try:
                        # Parse the JSON string into a Python dict
                        result_data = json.loads(result_json_string)
                        # Add to history (Lab 6d enhancement)
                        st.session_state.claim_history.insert(0, result_data)
                    except json.JSONDecodeError:
                        st.error("Failed to parse the response from the API.")
                        st.text(result_json_string) # Show raw text for debugging
        else:
            st.warning("Please enter a factual claim.")

    # --- 4. Display Results (Lab 6a & 6d) ---

    if st.session_state.claim_history:
        st.markdown("---")
        st.subheader("Latest Result")
        
        # Display the most recent result
        latest_result = st.session_state.claim_history[0]
        
        # Display as raw JSON (as required by Lab 6a)
        st.json(latest_result) 

        # --- Optional: Formatted Output (Lab 6d Enhancement) ---
        with st.expander("View Formatted Result (Lab 6d Enhancement)"):
            st.info(f"**Claim:** {latest_result.get('claim')}")
            st.success(f"**Verdict:** {latest_result.get('verdict')}")
            st.write(f"**Explanation:**\n{latest_result.get('explanation')}")
            
            st.write("**Sources:**")
            sources = latest_result.get('sources', [])
            if sources:
                for source in sources:
                    # Format sources as clickable Markdown links
                    st.markdown(f"- [{source.get('title')}]({source.get('url')})")
            else:
                st.write("No sources provided.")
        # --- End Formatted Output ---

        # Display history (Lab 6d enhancement)
        if len(st.session_state.claim_history) > 1:
            st.subheader("Checked Claims History")
            for item in st.session_state.claim_history[1:]:
                st.expander(f"**{item.get('verdict')}** - {item.get('claim')}")

    # --- 5. Reflection Section (Lab 6e) ---
    st.markdown("---")
    with st.expander("Lab 6 Reflection"):
        st.subheader("Reflection & Discussion")
        st.text_area(
            "How did the model’s reasoning feel different from a standard chat model?",
            key="lab6_reflection_1"
        )
        st.text_area(
            "Were the sources credible and diverse? Did you trust the verdict?",
            key="lab6_reflection_2"
        )
        st.text_area(
            "How does tool integration (web_search, json_schema) enhance trust and accuracy?",
            key="lab6_reflection_3"
        )

# --- Entry Point ---
# This allows the script to be run directly for testing
if __name__ == "__main__":
    main()