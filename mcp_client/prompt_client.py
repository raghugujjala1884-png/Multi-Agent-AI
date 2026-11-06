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

            # Discover prompts
            prompts = await session.list_prompts()

            print("Available Prompts:")

            for prompt in prompts.prompts:
                print(f"- {prompt.name}")

            # Get the explain_rag prompt
            result = await session.get_prompt(
                "explain_rag",
                arguments={
                    "audience": "beginner"
                }
            )

            print("\nPrompt Result:")

            for message in result.messages:
                print("\nRole:", message.role)
                print("Content:", message.content)


if __name__ == "__main__":
    asyncio.run(main())