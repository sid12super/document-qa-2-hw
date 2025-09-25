# lab5.py

import streamlit as st
import requests

def get_current_weather(location: str, api_key: str) -> dict:
    """
    Fetches the current weather for a given location using the OpenWeatherMap API.
    """
    # Standardize the location input
    if "," in location:
        location = location.split(",")[0].strip()

    # Construct the API request URL
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    request_url = f"{base_url}?q={location}&appid={api_key}"

    # Make the API call
    response = requests.get(request_url)

    # Check for a successful response
    if response.status_code == 200:
        data = response.json()
        
        # Extract and convert data from Kelvin to Celsius
        temp_celsius = data['main']['temp'] - 273.15
        feels_like_celsius = data['main']['feels_like'] - 273.15
        description = data['weather'][0]['description']
        humidity = data['main']['humidity']

        weather_data = {
            "location": location.capitalize(),
            "temperature_celsius": round(temp_celsius, 2),
            "feels_like_celsius": round(feels_like_celsius, 2),
            "humidity_percent": humidity,
            "description": description.title()
        }
        return weather_data
    else:
        # Return an error if the API call failed
        return {"error": f"API request failed for location '{location}'.", "details": response.json()}

def main():
    """
    The main function to run the Streamlit page for Lab 5.
    """
    st.title("☀️ Lab 5: Live Weather Check")
    
    # User input for location
    location_input = st.text_input("Enter a city name (e.g., Syracuse):", "Syracuse")

    # Button to trigger the API call
    if st.button("Get Current Weather"):
        if not location_input:
            st.warning("Please enter a location.")
        else:
            try:
                # Get the API key from secrets
                ow_api_key = st.secrets["OPENWEATHER_API_KEY"]
                
                # Show a spinner while fetching data
                with st.spinner(f"Fetching weather for {location_input}..."):
                    weather_info = get_current_weather(location_input, ow_api_key)
                
                # Display the results or an error
                if "error" in weather_info:
                    st.error(weather_info["error"])
                    st.json(weather_info["details"]) # Show detailed error from API
                else:
                    st.success(f"Weather in {weather_info['location']}:")
                    col1, col2 = st.columns(2)
                    col1.metric("Temperature", f"{weather_info['temperature_celsius']} °C")
                    col2.metric("Feels Like", f"{weather_info['feels_like_celsius']} °C")
                    st.metric("Conditions", f"{weather_info['description']}")
                    st.metric("Humidity", f"{weather_info['humidity_percent']}%")

            except (KeyError, FileNotFoundError):
                st.error("Error: OPENWEATHER_API_KEY not found in Streamlit secrets.")

# This block allows for direct testing of the script (optional)
if __name__ == "__main__":
    main()