# lab5.py

import streamlit as st
import requests
import openai
import json

# This function remains the same, as it will be our "tool".
def get_current_weather(location: str) -> dict:
    """
    Fetches the current weather for a given location using the OpenWeatherMap API.
    Note: The API key is retrieved inside this function to keep it self-contained.
    """
    try:
        api_key = st.secrets["OPENWEATHER_API_KEY"]
    except (KeyError, FileNotFoundError):
        return {"error": "OpenWeather API key not found in secrets."}
        
    if "," in location:
        location = location.split(",")[0].strip()

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    request_url = f"{base_url}?q={location}&appid={api_key}"
    response = requests.get(request_url)

    if response.status_code == 200:
        data = response.json()
        temp_celsius = data['main']['temp'] - 273.15
        feels_like_celsius = data['main']['feels_like'] - 273.15
        description = data['weather'][0]['description']
        
        return {
            "location": location.capitalize(),
            "temperature_celsius": round(temp_celsius, 2),
            "feels_like_celsius": round(feels_like_celsius, 2),
            "description": description.title()
        }
    else:
        return {"error": f"API request failed for location '{location}'."}

def run_conversation(user_prompt: str, client: openai.OpenAI) -> str:
    """
    Runs the two-step OpenAI call: first to decide to use a tool,
    and second to generate a suggestion based on the tool's output.
    """
    # Step 1: Define the function tool for the model
    tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit of temperature, either celsius or fahrenheit.",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"],
            },
        },
    }
]

    # Step 2: Send the user prompt to the model and let it decide to use the tool
    messages = [
        {"role": "system", "content": "You are a helpful assistant that suggests clothing based on the weather. If the user does not specify a location, default to Syracuse, NY."},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o", # Or any other suitable model
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Step 3: Check if the model wants to call the tool
    if tool_calls:
        # Step 4: Execute the function and get the weather data
        available_functions = {"get_current_weather": get_current_weather}
        
        # Note: For simplicity, we handle one tool call. A more complex app could loop over multiple.
        function_to_call = available_functions[tool_calls[0].function.name]
        function_args = json.loads(tool_calls[0].function.arguments)
        
        with st.spinner(f"Getting weather for {function_args.get('location')}..."):
            function_response = function_to_call(location=function_args.get("location"))

        # Step 5: Send the weather info back to the model in a second call to get the final suggestion
        messages.append(response_message)  # Add the assistant's turn with the tool call
        messages.append(
            {
                "tool_call_id": tool_calls[0].id,
                "role": "tool",
                "name": "get_current_weather",
                "content": json.dumps(function_response),
            }
        )
        
        # Add a final prompt to explicitly ask for the clothing suggestion
        messages.append({"role": "user", "content": "Based on this weather information, please provide suggestions on appropriate clothes to wear today."})
        
        with st.spinner("Thinking of a perfect outfit..."):
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
        return second_response.choices[0].message.content
    else:
        # If the model decides not to use the tool, return its direct response
        return response_message.content

def main():
    """
    The main function to run the Streamlit page for Lab 5.
    """
    st.title("👕 AI Clothing Advisor")
    
    try:
        # Set up OpenAI client from secrets
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        user_input = st.text_input("Ask for clothing advice (e.g., 'What should I wear in London?' or 'What's good for today?')", "What should I wear today?")
        
        if st.button("Get Suggestion"):
            if not user_input:
                st.warning("Please enter a question.")
            else:
                suggestion = run_conversation(user_input, client)
                st.markdown(suggestion)

    except (KeyError, FileNotFoundError):
        st.error("Error: Ensure both OPENAI_API_KEY and OPENWEATHER_API_KEY are in your Streamlit secrets.")

if __name__ == "__main__":
    main()