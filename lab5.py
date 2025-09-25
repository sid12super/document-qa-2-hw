# lab5.py

import streamlit as st
import requests
import openai
import json

# --- 1. The Local Tool ---
# This function is called by our Python code after the LLM decides to use it.
def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """
    Fetches the current weather and formats it, handling different units.
    """
    try:
        api_key = st.secrets["OPENWEATHER_API_KEY"]
    except KeyError:
        return {"error": "OpenWeather API key not found in secrets."}

    location_query = location.split(",")[0].strip()
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    request_url = f"{base_url}?q={location_query}&appid={api_key}"
    
    response = requests.get(request_url)
    if response.status_code != 200:
        return {"error": f"API request failed for location '{location}'."}

    data = response.json()
    temp_kelvin = data['main']['temp']
    description = data['weather'][0]['description']

    if unit.lower() == "fahrenheit":
        temp = (temp_kelvin - 273.15) * 9/5 + 32
        temp_unit = "Fahrenheit"
    else: # Default to Celsius
        temp = temp_kelvin - 273.15
        temp_unit = "Celsius"

    return {
        "location": location.capitalize(),
        "temperature": f"{round(temp, 2)}° {temp_unit}",
        "description": description.title()
    }

# --- 2. OpenAI-Specific Conversation Handler ---
def run_openai_conversation(user_prompt: str, client: openai.OpenAI, tools: list, system_prompt: str, model_name: str) -> str:
    """
    Runs the two-step OpenAI tool-calling conversation.
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    # First API call to decide if a tool should be used
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    response_message = response.choices[0].message

    # Check if the model wants to call our function
    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        
        # Call the actual local function
        function_response = get_current_weather(**function_args)
        
        # Append the history with the tool call and its response
        messages.append(response_message)
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": "get_current_weather",
            "content": json.dumps(function_response)
        })
        
        # Second API call to get the final, natural language response
        second_response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return second_response.choices[0].message.content
    else:
        # If no tool is needed, return the model's direct response
        return response_message.content

# --- 3. The Main Streamlit App ---
def main():
    st.title("🌦️ OpenAI Weather Assistant")
    st.write("Ask a weather-related question and get suggestions from your chosen AI model.")

    # --- Sidebar for OpenAI Model Selection ---
    with st.sidebar:
        st.header("Configuration")
        selected_model = st.selectbox(
            "Choose an OpenAI Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        )

    # --- Tool and System Prompt Definitions ---
    tool_schema = {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g., San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature"}
            },
            "required": ["location"]
        }
    }
    
    system_prompt = (
        "You are a helpful weather assistant. You provide suggestions and answer questions based ONLY on weather data. "
        "If the user asks about any topic other than weather (e.g., history, math, news), you must politely refuse to answer "
        "by stating you are only designed for weather-related questions. When asked for weather, use the provided tools. "
        "If no location is given, default to Syracuse, NY."
    )

    # --- Main App Logic ---
    user_input = st.text_input("Ask a question (e.g., 'Is it t-shirt weather in Paris today?')", "What should I wear in Syracuse today?")

    if st.button("Get Suggestion"):
        if not user_input:
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner(f"Contacting {selected_model}..."):
                client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                tools_list = [{"type": "function", "function": tool_schema}]
                
                result = run_openai_conversation(
                    user_prompt=user_input,
                    client=client,
                    tools=tools_list,
                    system_prompt=system_prompt,
                    model_name=selected_model
                )
                st.markdown(result)

        except KeyError as e:
            st.error(f"API Key Error: Please make sure `{e.args[0]}` is set in your Streamlit secrets.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()