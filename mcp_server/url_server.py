import httpx

from bs4 import BeautifulSoup
from mcp.server.mcpserver import MCPServer


mcp = MCPServer("URL Content Server")


@mcp.tool()
async def get_url_content(url: str) -> dict:
    """
    Fetch a webpage and return its title and readable text content.
    """

    async with httpx.AsyncClient() as client:

        response = await client.get(url)

        response.raise_for_status()

        html = response.text

        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = soup.title.string if soup.title else "No title"

        # Extract clean text
        text = soup.get_text(separator="\n", strip=True)

        return {
            "title": title,
            "content": text
        }