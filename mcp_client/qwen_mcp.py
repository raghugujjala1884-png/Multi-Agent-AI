import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage


server_params = StdioServerParameters(
    command="uv",
    args=[
        "run",
        "mcp",
        "run",
        "mcp_server/server.py"
    ],
)


llm = ChatOllama(
    model="qwen3:4b"
)


async def main():

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # Discover MCP tools
            mcp_tools = await session.list_tools()

            print("MCP Tools:")

            for tool in mcp_tools.tools:
                print(f"- {tool.name}")

            # Convert MCP tools to LangChain tools
            langchain_tools = []

            for tool in mcp_tools.tools:

                async def mcp_tool_function(
                    tool_name=tool.name,
                    **kwargs
                ):
                    result = await session.call_tool(
                        tool_name,
                        arguments=kwargs
                    )

                    return result.structured_content

                langchain_tool = StructuredTool.from_function(
                    coroutine=mcp_tool_function,
                    name=tool.name,
                    description=tool.description or "",
                )

                langchain_tools.append(langchain_tool)

            # Give MCP tools to Qwen
            llm_with_tools = llm.bind_tools(
                langchain_tools
            )

            print("\n" + "=" * 50)
            print("MCP + Qwen3 Chatbot")
            print("Type 'exit' to quit")
            print("=" * 50)

            while True:

                question = input("\nYou: ").strip()

                if question.lower() == "exit":
                    print("Goodbye!")
                    break

                if not question:
                    continue

                # First ask Qwen whether a tool is needed
                # response = llm_with_tools.invoke(question)
                messages = [
                    SystemMessage(
                        content=(
                            "You are a helpful AI assistant. "
                            "Answer general knowledge questions normally using your own knowledge. "
                            "Use the available MCP tools when the user asks for arithmetic calculations. "
                            "Do not claim that you cannot answer general questions just because the MCP tools "
                            "are arithmetic tools."
                        )
                    ),
                    ("user", question),
                ]

                response = llm_with_tools.invoke(messages)

                # If Qwen selected tools
                if response.tool_calls:

                    tool_messages = []

                    for tool_call in response.tool_calls:

                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_call_id = tool_call["id"]

                        print(
                            f"\n[Qwen selected MCP tool: {tool_name}]"
                        )

                        print(
                            f"[Arguments: {tool_args}]"
                        )

                        # Execute through MCP
                        result = await session.call_tool(
                            tool_name,
                            arguments=tool_args
                        )

                        print(
                            f"[MCP result: "
                            f"{result.structured_content}]"
                        )

                        tool_messages.append(
                            ToolMessage(
                                content=str(
                                    result.structured_content
                                ),
                                tool_call_id=tool_call_id
                            )
                        )

                    # Give MCP result back to Qwen
                    messages = [
                        {
                            "role": "user",
                            "content": question
                        },
                        response,
                        *tool_messages
                    ]

                    final_response = llm.invoke(messages)

                    print(f"\nAI: {final_response.content}")

                else:

                    # No MCP tool required
                    print(f"\nAI: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())