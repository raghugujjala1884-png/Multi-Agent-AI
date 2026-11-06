import httpx

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Weather Server")


@mcp.tool()
async def get_current_weather(city: str) -> dict:
    """
    Get the current weather for a city.
    """

    async with httpx.AsyncClient() as client:

        # Step 1: Get latitude and longitude
        geocoding_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
            f"?name={city}&count=1"
        )

        geo_response = await client.get(geocoding_url)
        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if "results" not in geo_data:
            return {
                "error": f"City '{city}' was not found."
            }

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # Step 2: Get current weather
        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )

        weather_response = await client.get(weather_url)
        weather_response.raise_for_status()

        weather_data = weather_response.json()

        return {
            "city": location["name"],
            "country": location.get("country"),
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": weather_data["current"]
        }


if __name__ == "__main__":
    mcp.run()