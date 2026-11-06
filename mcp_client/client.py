import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


server_params = StdioServerParameters(
    command="uv",
    args=[
        "run",
        "mcp",
        "run",
        "mcp_server/server.py"
    ],
)


async def main():

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # Discover tools
            tools = await session.list_tools()

            print("Available tools:")

            for tool in tools.tools:
                print(f"- {tool.name}")

            # Ask user which tool to call
            tool_name = input("\nEnter tool name: ").strip()

            # Call selected tool
            if tool_name == "add":
                result = await session.call_tool(
                    tool_name,
                    arguments={
                        "a": 10,
                        "b": 20
                    }
                )

            elif tool_name == "subtract":
                result = await session.call_tool(
                    tool_name,
                    arguments={
                        "a": 20,
                        "b": 10
                    }
                )

            elif tool_name == "multiply":
                result = await session.call_tool(
                    tool_name,
                    arguments={
                        "a": 10,
                        "b": 20
                    }
                )

            elif tool_name == "divide":
                result = await session.call_tool(
                    tool_name,
                    arguments={
                        "a": 20,
                        "b": 10
                    }
                )

            else:
                print("Unknown tool.")
                return

            print("\nTool result:")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())