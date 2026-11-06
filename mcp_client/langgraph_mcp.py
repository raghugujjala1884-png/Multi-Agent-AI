import asyncio
from typing import TypedDict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    messages: list


llm = ChatOllama(
    model="qwen3:4b"
)


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

            # -----------------------------
            # 1. Discover MCP tools
            # -----------------------------

            mcp_tools = await session.list_tools()

            print("MCP Tools:")

            for tool in mcp_tools.tools:
                print(f"- {tool.name}")

            # -----------------------------
            # 2. Convert MCP tools
            #    to LangChain tools
            # -----------------------------

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

                langchain_tools.append(
                    langchain_tool
                )

            # -----------------------------
            # 3. Give tools to Qwen
            # -----------------------------

            llm_with_tools = llm.bind_tools(
                langchain_tools
            )

            # -----------------------------
            # 4. Chatbot node
            # -----------------------------

            async def chatbot(state: State):

                response = await llm_with_tools.ainvoke(
                    state["messages"]
                )

                return {
                    "messages": (
                        state["messages"]
                        + [response]
                    )
                }

            # -----------------------------
            # 5. MCP tool execution node
            # -----------------------------

            async def tools_node(state: State):

                messages = state["messages"]

                last_message = messages[-1]

                tool_messages = []

                for tool_call in last_message.tool_calls:

                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    print(
                        f"\nCalling MCP tool: {tool_name}"
                    )

                    print(
                        f"Arguments: {tool_args}"
                    )

                    # Actual MCP call
                    result = await session.call_tool(
                        tool_name,
                        arguments=tool_args
                    )

                    # print(
                    #     f"MCP result: "
                    #     f"{result.structured_content}"
                    # )

                    # tool_messages.append(
                    #     ToolMessage(
                    #         content=str(
                    #             result.structured_content
                    #         ),
                    #         tool_call_id=tool_call_id
                    #     )
                    # )

                    if result.structured_content is not None:

                        tool_result = result.structured_content

                    else:

                        tool_result = "\n".join(
                            item.text
                            for item in result.content
                            if hasattr(item, "text")
                        )

                    print(
                        f"MCP result: {tool_result}"
                    )

                    tool_messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call_id
                        )
                    )

                return {
                    "messages": (
                        messages
                        + tool_messages
                    )
                }

            # -----------------------------
            # 6. Routing function
            # -----------------------------

            def route_after_chatbot(state: State):

                last_message = state["messages"][-1]

                if last_message.tool_calls:
                    return "tools"

                return END

            # -----------------------------
            # 7. Build LangGraph
            # -----------------------------

            builder = StateGraph(State)

            builder.add_node(
                "chatbot",
                chatbot
            )

            builder.add_node(
                "tools",
                tools_node
            )

            builder.add_edge(
                START,
                "chatbot"
            )

            builder.add_conditional_edges(
                "chatbot",
                route_after_chatbot,
                {
                    "tools": "tools",
                    END: END,
                }
            )

            builder.add_edge(
                "tools",
                "chatbot"
            )

            graph = builder.compile()

            # -----------------------------
            # 8. Run the graph
            # -----------------------------

            result = await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            # content="What is 25 multiplied by 4?"?
                            content="what is 100 divided by 0?"
                            
                        )
                    ]
                }
            )

            # -----------------------------
            # 9. Print final conversation
            # -----------------------------

            print("\n" + "=" * 50)
            print("FINAL CONVERSATION")
            print("=" * 50)

            for message in result["messages"]:

                print(
                    f"\n{type(message).__name__}:"
                )

                if message.content:
                    print(message.content)

                if hasattr(message, "tool_calls"):
                    if message.tool_calls:
                        print(
                            "Tool calls:",
                            message.tool_calls
                        )


if __name__ == "__main__":
    asyncio.run(main())