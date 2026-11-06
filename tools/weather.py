def get_weather(city: str):
    """Get the current weather for a city."""

    weather_data = {
        "hyderabad": "Sunny, 32°C",
        "bangalore": "Cloudy, 25°C",
        "ananthapur": "Sunny, 34°C",
        "chennai": "Hot and humid, 35°C",
    }

    city = city.strip().lower()

    if city not in weather_data:
        return {
            "success": False,
            "error": f"Weather information is not available for {city}."
        }

    return {
        "success": True,
        "city": city,
        "weather": weather_data[city]
    }