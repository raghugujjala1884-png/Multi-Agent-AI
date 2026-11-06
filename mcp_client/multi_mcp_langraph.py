import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():

    client = MultiServerMCPClient(
        {
            "calculator": {
                "command": "python",
                "args": ["mcp_server/server.py"],
                "transport": "stdio",
            },

            "weather": {
                "command": "python",
                "args": ["mcp_server/weather_server.py"],
                "transport": "stdio",
            },

            "documents": {
                "command": "python",
                "args": ["mcp_server/file_server.py"],
                "transport": "stdio",
            },

            "database": {
                "command": "python",
                "args": ["mcp_server/db_server.py"],
                "transport": "stdio",
            },
        }
    )

    # Load tools from all MCP servers
    tools = await client.get_tools()

    print("\nALL MCP TOOLS\n")
    print("=" * 50)

    for tool in tools:
        print(f"- {tool.name}")

    print("=" * 50)
    print(f"\nTotal Tools: {len(tools)}")


if __name__ == "__main__":
    asyncio.run(main())