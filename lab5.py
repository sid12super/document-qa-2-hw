# lab5.py

import streamlit as st
import requests
import openai
import anthropic
import google.generativeai as genai
import json

# --- 1. The Local Tool ---
# This function is called by our Python code after an LLM decides to use it.
def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """
    Fetches the current weather and formats it, handling different units.
    """
    try:
        api_key = st.secrets["OPENWEATHER_API_KEY"]
    except KeyError:
        return {"error": "OpenWeather API key not found in secrets."}

    # Standardize location and construct URL
    location_query = location.split(",")[0].strip()
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    request_url = f"{base_url}?q={location_query}&appid={api_key}"
    
    response = requests.get(request_url)
    if response.status_code != 200:
        return {"error": f"API request failed for location '{location}'."}

    data = response.json()
    temp_kelvin = data['main']['temp']
    description = data['weather'][0]['description']

    # Convert temperature based on the desired unit
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

# --- 2. Vendor-Specific Conversation Handlers ---

## OpenAI Handler
def run_openai_conversation(user_prompt: str, client: openai.OpenAI, tools: list, system_prompt: str) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    response = client.chat.completions.create(model="gpt-5-nano", messages=messages, tools=tools, tool_choice="auto")
    response_message = response.choices[0].message

    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        function_response = get_current_weather(**function_args)
        
        messages.append(response_message)
        messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": "get_current_weather", "content": json.dumps(function_response)})
        
        second_response = client.chat.completions.create(model="gpt-5-nano", messages=messages)
        return second_response.choices[0].message.content
    else:
        return response_message.content

## Anthropic (Claude) Handler
def run_anthropic_conversation(user_prompt: str, client: anthropic.Anthropic, tools: list, system_prompt: str) -> str:
    response = client.messages.create(
        model="claude-4-sonnet-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=tools,
        tool_choice={"type": "auto"}
    )

    if response.stop_reason == "tool_use":
        tool_call = next(block for block in response.content if block.type == "tool_use")
        function_args = tool_call.input
        function_response = get_current_weather(**function_args)
        
        # Construct the conversation history for the second call
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": json.dumps(function_response)
                }]
            }
        ]
        
        second_response = client.messages.create(model="claude-4-sonnet-20250514", max_tokens=1024, system=system_prompt, messages=messages)
        return second_response.content[0].text
    else:
        return response.content[0].text

## Google (Gemini) Handler
def run_google_conversation(user_prompt: str, client: genai.GenerativeModel, system_prompt: str) -> str:
    chat = client.start_chat()
    # Note: Gemini's system prompt is handled differently
    full_prompt = f"{system_prompt}\n\nUSER QUESTION: {user_prompt}"
    response = chat.send_message(full_prompt)
    
    try:
        function_call = response.candidates[0].content.parts[0].function_call
        if function_call.name == "get_current_weather":
            args = {key: value for key, value in function_call.args.items()}
            function_response = get_current_weather(**args)
            
            # Send the tool response back to the model
            second_response = chat.send_message(
                genai.types.Content(
                    parts=[genai.types.Part(
                        function_response=genai.types.FunctionResponse(name="get_current_weather", response=function_response)
                    )]
                )
            )
            return second_response.candidates[0].content.parts[0].text
    except (IndexError, AttributeError):
         # This happens if the model doesn't make a function call
        return response.candidates[0].content.parts[0].text

# --- 3. The Main Streamlit App ---

def main():
    st.title("🌦️ Universal Weather Assistant")
    st.write("Ask a weather-related question and get suggestions from your chosen AI.")

    # --- Sidebar for LLM Selection ---
    with st.sidebar:
        st.header("Configuration")
        vendor = st.selectbox("Choose AI Vendor", ["OpenAI", "Anthropic", "Google"])
        # In a real app, you might have another selectbox for model, but we'll hardcode them for simplicity.

    # --- Tool and System Prompt Definitions (shared across vendors) ---
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

        result = ""
        try:
            with st.spinner(f"Contacting {vendor}..."):
                if vendor == "OpenAI":
                    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    result = run_openai_conversation(user_input, client, [{"type": "function", "function": tool_schema}], system_prompt)
                
                elif vendor == "Anthropic":
                    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                    result = run_anthropic_conversation(user_input, client, [tool_schema], system_prompt)
                
                elif vendor == "Google":
                    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                    model = genai.GenerativeModel(model_name="gemini-2.5-pro", tools=[tool_schema])
                    result = run_google_conversation(user_input, model, system_prompt)

            st.markdown(result)

        except KeyError as e:
            st.error(f"API Key Error: Please make sure `{e.args[0]}` is set in your Streamlit secrets.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()