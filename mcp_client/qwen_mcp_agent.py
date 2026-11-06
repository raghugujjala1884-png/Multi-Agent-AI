import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


async def main():

    # --------------------------------------------------
    # 1. CONNECT TO ALL MCP SERVERS
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 2. LOAD MCP TOOLS
    # --------------------------------------------------

    tools = await client.get_tools()

    print("\n" + "=" * 60)
    print("MCP TOOLS LOADED")
    print("=" * 60)

    for tool in tools:
        print(f"- {tool.name}")

    print(f"\nTotal Tools: {len(tools)}")

    # --------------------------------------------------
    # 3. CONNECT QWEN3 MODEL
    # --------------------------------------------------

    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
    )

    # --------------------------------------------------
    # 4. BIND MCP TOOLS TO QWEN
    # --------------------------------------------------

    llm_with_tools = llm.bind_tools(tools)

    # --------------------------------------------------
    # 5. TEST MODEL
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TESTING QWEN + MCP")
    print("=" * 60)

    response = await llm_with_tools.ainvoke(
        "What is 25 multiplied by 4?"
    )

    print("\nMODEL RESPONSE:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())