import httpx

from mcp.server.mcpserver import MCPServer


mcp = MCPServer("External API Server")


@mcp.tool()
async def get_external_user(user_id: int) -> dict:
    """Get user information from an external REST API."""

    url = f"https://jsonplaceholder.typicode.com/users/{user_id}"

    async with httpx.AsyncClient() as client:

        response = await client.get(url)

        response.raise_for_status()

        return response.json()