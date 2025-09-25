# lab5.py

import streamlit as st
import requests

def get_current_weather(location: str, api_key: str) -> dict:
    """
    Fetches the current weather for a given location using the OpenWeatherMap API.

    Args:
        location (str): The name of the city (e.g., "Syracuse" or "Syracuse, NY").
        api_key (str): Your OpenWeatherMap API key.

    Returns:
        dict: A dictionary containing cleaned weather data or an error message.
    """
    # Standardize the location input by taking the part before a comma
    if "," in location:
        location = location.split(",")[0].strip()

    # Construct the API request URL
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    request_url = f"{base_url}?q={location}&appid={api_key}"

    # Make the API call
    response = requests.get(request_url)

    # Check if the request was successful (HTTP Status Code 200)
    if response.status_code == 200:
        data = response.json()

        # Extract relevant weather information
        # Note: Temperatures are converted from Kelvin to Celsius
        temp_celsius = data['main']['temp'] - 273.15
        feels_like_celsius = data['main']['feels_like'] - 273.15
        temp_min_celsius = data['main']['temp_min'] - 273.15
        temp_max_celsius = data['main']['temp_max'] - 273.15
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']

        # Structure and return the final weather data
        weather_data = {
            "location": location.capitalize(),
            "temperature_celsius": round(temp_celsius, 2),
            "feels_like_celsius": round(feels_like_celsius, 2),
            "temp_min_celsius": round(temp_min_celsius, 2),
            "temp_max_celsius": round(temp_max_celsius, 2),
            "humidity_percent": humidity,
            "description": description.title()
        }
        return weather_data
    else:
        # Return an error message if the API call failed
        return {"error": f"API request failed with status code {response.status_code}", "details": response.json()}

# This block allows for direct testing of the script
if __name__ == "__main__":
    # This part assumes you have your secrets.toml file configured
    # and you run this script in an environment where Streamlit can access it.
    try:
        # Access the API key from Streamlit's secrets management
        ow_api_key = st.secrets["OPENWEATHER_API_KEY"]
        
        # Define a test location
        test_location = "Syracuse, NY"
        
        # Call the function to get weather data
        weather_info = get_current_weather(test_location, ow_api_key)
        
        # Print the results
        print(f"Weather information for {test_location}:")
        for key, value in weather_info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
            
    except (KeyError, FileNotFoundError):
        print("Error: Could not find OPENWEATHER_API_KEY in your Streamlit secrets.")
        print("Please ensure your .streamlit/secrets.toml file is correctly configured.")