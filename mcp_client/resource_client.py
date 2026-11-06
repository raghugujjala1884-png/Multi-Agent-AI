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

            # List available resources
            resources = await session.list_resources()

            print("Available Resources:")

            for resource in resources.resources:
                print(
                    f"- {resource.uri}"
                )

            # Read the RAG resource
            result = await session.read_resource(
                "info://rag"
            )

            print("\nResource Content:")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())