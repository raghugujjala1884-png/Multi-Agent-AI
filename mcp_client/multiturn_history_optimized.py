import sys
import asyncio
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_mcp_adapters.client import MultiServerMCPClient


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ============================================================
# TOOL ROUTING
# ============================================================

def route_tools(question: str, tools: list):

    question_lower = question.lower()

    database_keywords = [
        "employee",
        "employee id",
        "department",
        "designation",
        "salary",
    ]

    weather_keywords = [
        "weather",
        "temperature",
        "humidity",
        "wind",
    ]

    document_keywords = [
        "policy",
        "document",
        "resume",
        "pdf",
        "file",
    ]

    calculator_keywords = [
        "calculate",
        "add",
        "subtract",
        "multiply",
        "divide",
        "+",
        "-",
        "*",
        "/",
    ]

    if any(keyword in question_lower for keyword in database_keywords):

        selected_names = {
            "get_employee",
            "get_employees_by_department",
            "search_employee_by_name",
        }

    elif any(keyword in question_lower for keyword in weather_keywords):

        selected_names = {
            "get_current_weather",
        }

    elif any(keyword in question_lower for keyword in document_keywords):

        selected_names = {
            "get_company_policy",
            "get_all_csv_employees",
            "search_csv_employee",
            "get_csv_employees_by_department",
            "get_resume_content",
            "get_pdf_content",
            "list_documents",
        }

    elif any(keyword in question_lower for keyword in calculator_keywords):

        selected_names = {
            "add",
            "subtract",
            "multiply",
            "divide",
        }

    else:

        selected_names = {
            tool.name
            for tool in tools
        }

    return [
        tool
        for tool in tools
        if tool.name in selected_names
    ]


# ============================================================
# HISTORY MANAGEMENT
# ============================================================

def get_recent_history(
    conversation: list[AnyMessage],
    current_question: str,
) -> list[AnyMessage]:

    """
    Keep:
        1. Previous complete conversation turn
        2. Current user question

    A tool-based turn normally contains:

        HumanMessage
        AIMessage (tool call)
        ToolMessage
        AIMessage (final answer)
    """

    # Current question is already part of conversation.
    current_question_message = conversation[-1]

    # If this is the first turn, only send current question.
    if len(conversation) <= 4:

        return [
            current_question_message
        ]


    # --------------------------------------------------------
    # Find HumanMessage positions.
    # --------------------------------------------------------

    human_positions = [
        index
        for index, message in enumerate(conversation)
        if isinstance(message, HumanMessage)
    ]


    # Need at least two user questions to get
    # previous turn + current question.
    if len(human_positions) < 2:

        return [
            current_question_message
        ]


    # --------------------------------------------------------
    # Position of previous user question.
    # --------------------------------------------------------

    previous_human_index = human_positions[-2]

    # --------------------------------------------------------
    # Previous complete turn starts from previous HumanMessage.
    #
    # Current question is kept separately.
    # --------------------------------------------------------

    previous_turn = conversation[
        previous_human_index:
        -1
    ]

    return previous_turn + [
        current_question_message
    ]


# ============================================================
# MCP CLIENT
# ============================================================

async def main():

    print("\n========== OPTIMIZED MULTI-TURN HISTORY TEST ==========\n")

    total_input_tokens = 0
    total_output_tokens = 0


    # --------------------------------------------------------
    # MCP SERVERS
    # --------------------------------------------------------

    client = MultiServerMCPClient({

        "calculator": {
            "command": sys.executable,
            "args": [
                str(PROJECT_ROOT / "mcp_server" / "server.py")
            ],
            "transport": "stdio",
        },

        "weather": {
            "command": sys.executable,
            "args": [
                str(PROJECT_ROOT / "mcp_server" / "weather_server.py")
            ],
            "transport": "stdio",
        },

        "documents": {
            "command": sys.executable,
            "args": [
                str(PROJECT_ROOT / "mcp_server" / "file_server.py")
            ],
            "transport": "stdio",
        },

        "database": {
            "command": sys.executable,
            "args": [
                str(PROJECT_ROOT / "mcp_server" / "db_server.py")
            ],
            "transport": "stdio",
        },
    })


    # --------------------------------------------------------
    # LOAD TOOLS
    # --------------------------------------------------------

    tools = await client.get_tools()

    print(
        f"Total MCP tools available: {len(tools)}"
    )


    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        num_predict=1024,
        think=False,
    )


    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    questions = [
        "Find the employee with ID EMP001.",
        "Now find the employee with ID EMP003.",
        "Finally find the employee with ID EMP005.",
    ]


    # --------------------------------------------------------
    # FULL APPLICATION HISTORY
    #
    # IMPORTANT:
    # We keep everything here.
    # Only the context sent to the LLM is trimmed.
    # --------------------------------------------------------

    conversation = []


    # ========================================================
    # MULTI-TURN LOOP
    # ========================================================

    for turn_number, question in enumerate(
        questions,
        start=1
    ):

        print("\n" + "=" * 60)
        print(f"TURN {turn_number}")
        print("=" * 60)

        print(
            f"User: {question}"
        )


        # ----------------------------------------------------
        # Add current question to FULL history.
        # ----------------------------------------------------

        conversation.append(
            HumanMessage(
                content=question
            )
        )


        print(
            f"Full history messages: "
            f"{len(conversation)}"
        )


        # ----------------------------------------------------
        # CREATE OPTIMIZED LLM CONTEXT
        # ----------------------------------------------------

        llm_context = get_recent_history(
            conversation,
            question,
        )


        print(
            f"Messages sent to LLM: "
            f"{len(llm_context)}"
        )


        # ----------------------------------------------------
        # TOOL ROUTING
        # ----------------------------------------------------

        selected_tools = route_tools(
            question,
            tools
        )


        print(
            "\nTools sent to first LLM call:"
        )

        for tool in selected_tools:

            print(
                f"  - {tool.name}"
            )


        llm_with_tools = llm.bind_tools(
            selected_tools
        )


        # ====================================================
        # AGENT NODE
        # ====================================================

        async def agent(state):

            nonlocal total_input_tokens
            nonlocal total_output_tokens

            messages = state["messages"]


            # ------------------------------------------------
            # IMPORTANT:
            #
            # If this is the tool-result step, use the
            # current graph state because it contains:
            #
            # AI tool call
            # Tool result
            #
            # This prevents breaking the tool-call pair.
            # ------------------------------------------------

            if (
                messages
                and isinstance(
                    messages[-1],
                    ToolMessage
                )
            ):

                response = await llm.ainvoke(
                    messages
                )

                call_type = "FINAL ANSWER CALL"

            else:

                response = await llm_with_tools.ainvoke(
                    llm_context
                )

                call_type = "TOOL SELECTION CALL"


            # ------------------------------------------------
            # TOKEN USAGE
            # ------------------------------------------------

            metadata = response.response_metadata

            input_tokens = metadata.get(
                "prompt_eval_count",
                0
            )

            output_tokens = metadata.get(
                "eval_count",
                0
            )

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens


            print(
                f"\n{call_type}"
            )

            print(
                f"Input tokens : "
                f"{input_tokens}"
            )

            print(
                f"Output tokens: "
                f"{output_tokens}"
            )

            print(
                f"Total tokens : "
                f"{input_tokens + output_tokens}"
            )


            # ------------------------------------------------
            # TOOL CALL
            # ------------------------------------------------

            if response.tool_calls:

                print(
                    "\nTool selected:"
                )

                for tool_call in response.tool_calls:

                    print(
                        f"  {tool_call['name']}"
                    )

                    print(
                        f"  Arguments: "
                        f"{tool_call['args']}"
                    )


            return {
                "messages": [response]
            }


        # ====================================================
        # BUILD GRAPH
        # ====================================================

        tool_node = ToolNode(
            selected_tools
        )

        graph_builder = StateGraph(
            AgentState
        )

        graph_builder.add_node(
            "agent",
            agent
        )

        graph_builder.add_node(
            "tools",
            tool_node
        )

        graph_builder.add_edge(
            START,
            "agent"
        )

        graph_builder.add_conditional_edges(
            "agent",
            tools_condition
        )

        graph_builder.add_edge(
            "tools",
            "agent"
        )

        graph = graph_builder.compile()


        # ====================================================
        # RUN TURN
        # ====================================================

        result = await graph.ainvoke(
            {
                "messages": llm_context
            }
        )


        # ----------------------------------------------------
        # Add the NEW messages produced by this turn to
        # the FULL application history.
        #
        # We do NOT replace full history with trimmed context.
        # ----------------------------------------------------

        new_messages = result["messages"][
            len(llm_context):
        ]

        conversation.extend(
            new_messages
        )


        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        final_message = conversation[-1]

        print(
            "\nAssistant:"
        )

        print(
            final_message.content
        )


        print(
            f"\nFull history after turn "
            f"{turn_number}: "
            f"{len(conversation)} messages"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("OPTIMIZED HISTORY SUMMARY")
    print("=" * 60)

    print(
        f"Total input tokens : "
        f"{total_input_tokens}"
    )

    print(
        f"Total output tokens: "
        f"{total_output_tokens}"
    )

    print(
        f"Total tokens       : "
        f"{total_input_tokens + total_output_tokens}"
    )

    print(
        f"\nFinal full history: "
        f"{len(conversation)} messages"
    )


if __name__ == "__main__":
    asyncio.run(main())