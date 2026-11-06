import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main():

    async with streamable_http_client(
        "http://127.0.0.1:8000/mcp"
    ) as (read, write):

        async with ClientSession(
            read,
            write
        ) as session:

            # MCP handshake
            await session.initialize()

            # Discover tools
            tools = await session.list_tools()

            print("Available Tools:")

            for tool in tools.tools:
                print(f"- {tool.name}")

            # Call a tool
            result = await session.call_tool(
                "multiply",
                arguments={
                    "a": 25,
                    "b": 4
                }
            )

            print("\nTool Result:")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())