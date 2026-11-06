import asyncio
import json
import logging
import time

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import tools_condition

from tool_cache import ToolCache
from tracer import Trace
from cost_calculator import CostCalculator


# ==================================================
# CONFIGURATION
# ==================================================

MAX_TOOL_CALLS = 3
MODEL_NAME = "qwen3:4b"


# ==================================================
# LOGGING CONFIGURATION
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def clean_final_answer(answer: str) -> str:
    """
    Remove reasoning blocks and thinking tags from
    the final user-facing answer.
    """

    if not answer:
        return ""

    while "<think>" in answer and "</think>" in answer:
        start = answer.find("<think>")
        end = answer.find("</think>") + len("</think>")
        answer = answer[:start] + answer[end:]

    answer = answer.replace("<think>", "")
    answer = answer.replace("</think>", "")

    return answer.strip()


# ==================================================
# MAIN AGENT
# ==================================================

async def main(question: str):
    """
    Run the LangGraph MCP agent for one dynamic user question.

    Agent behavior:

    1. The LLM receives all available MCP tools.
    2. The LLM decides whether a tool is needed.
    3. If a suitable tool exists, it calls the tool.
    4. The tool result is passed to a final LLM with strict grounding rules.
    5. The final LLM converts the tool result into the user-facing answer
       without adding facts or changing values.
    6. If no tool is needed, the agent answers using its own knowledge.

    The returned dictionary contains observability information so
    we can manually verify what the agent actually did.
    """

    # ==================================================
    # 0. INITIALIZE OBSERVABILITY
    # ==================================================

    total_start = time.perf_counter()

    total_input_tokens = 0
    total_output_tokens = 0
    tool_call_count = 0

    tool_cache = ToolCache(ttl_seconds=300)

    CACHEABLE_TOOLS = {
        "get_employee",
        "get_employees_by_department",
        "search_employee_by_name",
    }

    trace = Trace()

    cost_calculator = CostCalculator(
        input_price_per_1m=1.0,
        output_price_per_1m=4.0
    )

    # Keep ALL selected tools and results instead of only the last one.
    selected_tool_calls = []
    executed_tool_results = []

    logger.info(
        "Agent execution started | Trace ID=%s",
        trace.trace_id
    )

    # ==================================================
    # 1. CONNECT TO MCP SERVERS
    # ==================================================

    client = MultiServerMCPClient(
        {
            "calculator": {
                "command": "python",
                "args": [
                    "mcp_server/server.py"
                ],
                "transport": "stdio",
            },
            "weather": {
                "command": "python",
                "args": [
                    "mcp_server/weather_server.py"
                ],
                "transport": "stdio",
            },
            "documents": {
                "command": "python",
                "args": [
                    "mcp_server/file_server.py"
                ],
                "transport": "stdio",
            },
            "database": {
                "command": "python",
                "args": [
                    "mcp_server/db_server.py"
                ],
                "transport": "stdio",
            },
        }
    )

    # ==================================================
    # 2. LOAD ALL MCP TOOLS
    # ==================================================

    # IMPORTANT:
    # We intentionally do NOT hard-code keyword based routing here.
    # The LLM receives all available tools and decides dynamically
    # whether one of them can answer the user's question.

    tools = await client.get_tools()

    print("\n" + "=" * 70)
    print("AVAILABLE MCP TOOLS")
    print("=" * 70)

    for tool in tools:
        print(f"- {tool.name}")

    print(f"\nTotal Tools: {len(tools)}")
    print("=" * 70)

    logger.info(
        "MCP tools loaded: %d",
        len(tools)
    )

    # ==================================================
    # 3. INITIALIZE QWEN3
    # ==================================================

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
        num_predict=256,
        reasoning=False
    )

    logger.info(
        "LLM initialized: %s",
        MODEL_NAME
    )

    # Final-answer LLM.
    # This LLM is used ONLY after a tool has returned a result.
    # It must answer strictly from the tool result and must not invent
    # or add information from its own knowledge.
    final_llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
        num_predict=256,
        reasoning=False
    )

    # Give the agent access to ALL MCP tools.
    llm_with_tools = llm.bind_tools(tools)

    # ==================================================
    # 4. AGENT NODE
    # ==================================================

    async def agent(state):
        nonlocal total_input_tokens
        nonlocal total_output_tokens
        nonlocal selected_tool_calls

        messages = state["messages"]

        system_message = SystemMessage(
            content=(
                "You are a tool-aware assistant.\n\n"

                "Your job is to answer the user's question accurately.\n\n"

                "TOOL RULES:\n"

                "1. Examine all available tools before deciding how to answer.\n"

                "2. If the user's question asks for information that can be "
                "looked up, retrieved, calculated, searched, or obtained by "
                "an available tool, you MUST call the appropriate tool.\n"

                "3. ALWAYS use lookup/search/database tools when the user asks "
                "for a specific record, employee, document, weather information, "
                "calculation, or other information that the tool is designed to provide.\n"

                "4. Do NOT decide whether the requested record exists before "
                "calling the tool. Let the tool determine whether the data exists.\n"

                "5. For example, if the user asks for employee EMP999 and "
                "get_employee is available, call get_employee with "
                "employee_id='EMP999'. The tool must determine whether EMP999 exists.\n"

                "6. Choose the most appropriate available tool.\n"

                "7. Extract the correct arguments from the user's question.\n"

                "8. Do NOT invent tool results.\n"

                "9. If no available tool is relevant to the user's question, "
                "do NOT call an unrelated tool. Answer using your own knowledge.\n"

                "10. If you use a tool, the application will execute it and "
                "generate a grounded final answer from the tool result.\n"

                "11. Do not call a tool merely because one exists; it must be "
                "relevant to the user's request.\n"

                "12. Do not call more tools than necessary."
            )
        )

        llm_messages = [
            system_message,
            *messages,
        ]

        llm_span = trace.start_span(
            "LLM agent decision"
        )

        try:
            response = await llm_with_tools.ainvoke(
                llm_messages
            )

        except Exception as exc:
            llm_span.end(
                status="error",
                metadata={
                    "error": str(exc)
                }
            )
            raise

        # ==================================================
        # TOKEN USAGE
        # ==================================================

        metadata = response.response_metadata or {}

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

        llm_span.end(
            status="success",
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        )

        print("\n" + "=" * 70)
        print("AGENT DECISION")
        print("=" * 70)
        print(f"Input tokens : {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Total tokens : {input_tokens + output_tokens}")

        # ==================================================
        # TOOL SELECTION
        # ==================================================

        if response.tool_calls:
            print("\nTool(s) selected by agent:")

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                arguments = tool_call.get("args", {})

                selected_tool_calls.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "tool_call_id": tool_call["id"],
                })

                print(f"  Tool      : {tool_name}")
                print(f"  Arguments : {arguments}")

        else:
            print("\nNo MCP tool selected.")
            print("Agent will answer using its own knowledge.")

            # --------------------------------------------------
            # NO-TOOL FINAL ANSWER
            # --------------------------------------------------
            # The tool-selection call is not used directly as the
            # user-facing answer. Qwen can include internal-style
            # reasoning while deciding whether a tool is needed.
            # Therefore, when no tool is selected, make a separate
            # final-answer call with strict output instructions.

            no_tool_system_message = SystemMessage(
                content=(
                    "You are the final answer generator.\n\n"

                    "Answer the user's question using your own knowledge.\n\n"

                    "IMPORTANT:\n"
                    "Return ONLY the final answer that should be shown to the user.\n\n"

                    "STRICT RULES:\n"
                    "1. Do NOT output reasoning, analysis, thinking, planning, or self-talk.\n"
                    "2. Do NOT describe what you are doing.\n"
                    "3. Do NOT mention MCP, tools, agents, prompts, instructions, or tool selection.\n"
                    "4. Do NOT say phrases such as 'the user is asking', 'I need to', "
                    "'I should', 'let me', 'wait', 'I think', or 'let's see'.\n"
                    "5. Do NOT output <think> or </think> tags.\n"
                    "6. Do NOT explain how you decided to answer the question.\n"
                    "7. Answer the user's question directly and clearly.\n"
                    "8. Do not invent specific facts when you are uncertain.\n"
                    "9. Return ONLY the final answer with no preamble.\n"
                )
            )

            final_messages = [
                no_tool_system_message,
                *messages,
            ]

            final_span = trace.start_span(
                "LLM final answer - no tool"
            )

            try:
                final_response = await final_llm.ainvoke(
                    final_messages
                )

                cleaned_content = clean_final_answer(final_response.content)

                final_response = AIMessage(
                    content=cleaned_content
                )

            except Exception as exc:
                final_span.end(
                    status="error",
                    metadata={
                        "error": str(exc)
                    }
                )
                raise

            final_metadata = final_response.response_metadata or {}

            final_input_tokens = final_metadata.get(
                "prompt_eval_count",
                0
            )

            final_output_tokens = final_metadata.get(
                "eval_count",
                0
            )

            total_input_tokens += final_input_tokens
            total_output_tokens += final_output_tokens

            final_span.end(
                status="success",
                metadata={
                    "input_tokens": final_input_tokens,
                    "output_tokens": final_output_tokens,
                    "total_tokens": final_input_tokens + final_output_tokens
                }
            )

            print("\nFinal answer generated from LLM knowledge.")
            print(
                f"Final answer tokens: {final_input_tokens + final_output_tokens}"
            )

            print("=" * 70)

            return {
                "messages": [final_response]
            }

        print("=" * 70)

        return {
            "messages": [response]
        }

    # ==================================================
    # 5. TOOL EXECUTION NODE
    # ==================================================

    async def execute_tools(state):
        nonlocal tool_call_count
        nonlocal executed_tool_results

        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage):
            return {
                "messages": []
            }

        tool_calls = last_message.tool_calls

        if not tool_calls:
            return {
                "messages": []
            }

        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            arguments = tool_call.get("args", {})

            tool_call_count += 1

            if tool_call_count > MAX_TOOL_CALLS:
                raise RuntimeError(
                    f"Maximum tool-call limit of "
                    f"{MAX_TOOL_CALLS} exceeded."
                )

            logger.info(
                "Tool execution started | Tool=%s | Arguments=%s",
                tool_name,
                arguments
            )

            # ==================================================
            # CACHE CHECK
            # ==================================================

            if tool_name in CACHEABLE_TOOLS:
                cache_span = trace.start_span(
                    "Cache lookup"
                )

                cached_result = tool_cache.get(
                    tool_name,
                    arguments
                )

                if cached_result is not None:
                    cache_span.end(
                        status="hit",
                        metadata={
                            "tool": tool_name,
                            "arguments": arguments
                        }
                    )

                    logger.info(
                        "CACHE HIT | Tool=%s | Arguments=%s",
                        tool_name,
                        arguments
                    )

                    executed_tool_results.append({
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": cached_result,
                        "source": "cache",
                    })

                    results.append(
                        ToolMessage(
                            content=str(cached_result),
                            tool_call_id=tool_call["id"],
                            name=tool_name
                        )
                    )

                    continue

                cache_span.end(
                    status="miss",
                    metadata={
                        "tool": tool_name,
                        "arguments": arguments
                    }
                )

                logger.info(
                    "CACHE MISS | Tool=%s | Arguments=%s",
                    tool_name,
                    arguments
                )

            # ==================================================
            # MCP TOOL EXECUTION
            # ==================================================

            tool_start = time.perf_counter()

            mcp_span = trace.start_span(
                f"MCP {tool_name}"
            )

            try:
                result = await selected_tool_map[tool_name].ainvoke(
                    arguments
                )

            except Exception as exc:
                mcp_span.end(
                    status="error",
                    metadata={
                        "tool": tool_name,
                        "arguments": arguments,
                        "error": str(exc)
                    }
                )

                logger.exception(
                    "Tool execution failed | Tool=%s",
                    tool_name
                )

                executed_tool_results.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": f"Error executing tool: {exc}",
                    "source": "mcp_error",
                })

                results.append(
                    ToolMessage(
                        content=f"Error executing tool: {exc}",
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    )
                )

                continue

            tool_latency = time.perf_counter() - tool_start

            mcp_span.end(
                status="success",
                metadata={
                    "tool": tool_name,
                    "arguments": arguments,
                    "latency": tool_latency
                }
            )

            executed_tool_results.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "source": "mcp",
            })

            logger.info(
                "Tool execution completed | Tool=%s | Latency=%.2f sec",
                tool_name,
                tool_latency
            )

            # ==================================================
            # STORE SUCCESSFUL RESULT IN CACHE
            # ==================================================

            if (
                tool_name in CACHEABLE_TOOLS
                and result not in (None, "", [], {})
            ):
                tool_cache.set(
                    tool_name,
                    arguments,
                    result
                )

                logger.info(
                    "CACHE STORED | Tool=%s | Arguments=%s",
                    tool_name,
                    arguments
                )

            results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_name
                )
            )

        return {
            "messages": results
        }

    # The tool map is created after tools are loaded.
    selected_tool_map = {
        tool.name: tool
        for tool in tools
    }

    # ==================================================
    # 6. GUARDRAIL CHECK
    # ==================================================

    def guardrail_check(state: MessagesState):
        guardrail_start = time.perf_counter()

        guardrail_span = trace.start_span(
            "Guardrail"
        )

        last_message = state["messages"][-1]

        # Maximum tool calls.
        if tool_call_count > MAX_TOOL_CALLS:
            logger.warning(
                "Tool-call limit reached | Calls=%s | Limit=%s",
                tool_call_count,
                MAX_TOOL_CALLS
            )

            guardrail_span.end(
                status="blocked",
                metadata={
                    "result": "LOOP_LIMIT"
                }
            )

            return "loop_limit"

        if not isinstance(last_message, ToolMessage):
            guardrail_span.end(
                status="success",
                metadata={
                    "result": "PASS"
                }
            )

            return "valid"

        # ==================================================
        # CHECK TOOL OUTPUT
        # ==================================================

        tool_content = last_message.content

        if isinstance(tool_content, list):
            tool_content = str(tool_content)

        tool_content = str(tool_content).strip()

        # Empty result.
        if tool_content == "[]":
            logger.warning(
                "Guardrail blocked empty tool result"
            )

            guardrail_span.end(
                status="blocked",
                metadata={
                    "result": "EMPTY"
                }
            )

            return "empty"

        # Error result.
        if "error" in tool_content.lower():
            logger.warning(
                "Guardrail blocked tool error"
            )

            guardrail_span.end(
                status="blocked",
                metadata={
                    "result": "ERROR"
                }
            )

            return "error"

        guardrail_end = time.perf_counter()

        logger.info(
            "Guardrail passed | Latency=%.5f sec",
            guardrail_end - guardrail_start
        )

        guardrail_span.end(
            status="success",
            metadata={
                "result": "PASS"
            }
        )

        return "valid"

    # ==================================================
    # 7. FINAL ANSWER NODE
    # ==================================================

    async def final_answer_node(state: MessagesState):
        """
        Generate the final user-facing answer after tool execution.

        The LLM is strictly grounded in the MCP tool result. It is NOT
        allowed to add facts from its own knowledge or change values.
        Its only job is to convert the structured tool result into a
        clear natural-language answer to the user's question.
        """

        nonlocal total_input_tokens
        nonlocal total_output_tokens

        final_system_message = SystemMessage(
            content=(
                "You are the final answer generator.\n\n"

                "The user asked a question and an MCP tool has returned the "
                "actual information needed to answer it.\n\n"

                "IMPORTANT:\n"
                "Return ONLY the final answer that should be shown to the user.\n\n"

                "STRICT RULES:\n"
                "1. Use ONLY information contained in the MCP tool result.\n"
                "2. Do NOT use general knowledge to add information.\n"
                "3. Do NOT invent, assume, estimate, or modify any value.\n"
                "4. Do NOT change names, IDs, numbers, dates, or other facts.\n"
                "5. Do NOT output reasoning, analysis, thinking, planning, or self-talk.\n"
                "6. Do NOT describe what you are doing.\n"
                "7. Do NOT mention MCP, tools, agents, prompts, instructions, "
                "or tool selection.\n"
                "8. Do NOT say phrases such as 'the user is asking', 'I need to', "
                "'I should', 'let me', 'wait', 'I think', or 'let's see'.\n"
                "9. Do NOT output <think> or </think> tags.\n"
                "10. Do NOT repeat the raw tool-call JSON unless the user asks for it.\n"
                "11. Give a concise, direct answer to the user's question.\n"
                "12. If the tool result contains multiple records, include the records "
                "relevant to the user's question.\n"
                "13. If the tool result does not contain enough information, clearly "
                "say that the available information does not contain enough information.\n"
                "14. Return ONLY the final answer with no explanation of your process.\n"
            )
        )

        llm_messages = [
            final_system_message,
            *state["messages"],
        ]

        final_span = trace.start_span(
            "LLM final answer"
        )

        try:
            response = await final_llm.ainvoke(
                llm_messages
            )

        except Exception as exc:
            final_span.end(
                status="error",
                metadata={
                    "error": str(exc)
                }
            )
            raise

        metadata = response.response_metadata or {}

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

        final_span.end(
            status="success",
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        )

        print("\n" + "=" * 70)
        print("FINAL ANSWER GENERATION")
        print("=" * 70)
        print(f"Input tokens : {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Total tokens : {input_tokens + output_tokens}")
        print("Answer generated from MCP tool result.")
        print("=" * 70)

        return {
            "messages": [response]
        }

    # ==================================================
    # 8. EMPTY RESULT RESPONSE
    # ==================================================

    def empty_response(state: MessagesState):
        logger.warning(
            "GUARDRAIL ACTIVATED | Reason: Tool returned no data."
        )

        return {
            "messages": [
                AIMessage(
                    content="No data was found for your request."
                )
            ]
        }

    # ==================================================
    # 9. ERROR RESPONSE
    # ==================================================

    def error_response(state: MessagesState):
        logger.warning(
            "GUARDRAIL ACTIVATED | Reason: Tool returned an error."
        )

        return {
            "messages": [
                AIMessage(
                    content=(
                        "The requested information "
                        "could not be retrieved."
                    )
                )
            ]
        }

    # ==================================================
    # 10. LOOP LIMIT RESPONSE
    # ==================================================

    def loop_limit_response(state: MessagesState):
        logger.warning(
            "LOOP CONTROL ACTIVATED | Maximum tool-call limit reached."
        )

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I stopped processing because the maximum "
                        "number of tool calls was reached."
                    )
                )
            ]
        }

    # ==================================================
    # 11. BUILD LANGGRAPH
    # ==================================================

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(
        "agent",
        agent
    )

    graph_builder.add_node(
        "tools",
        execute_tools
    )

    graph_builder.add_node(
        "final_answer",
        final_answer_node
    )

    graph_builder.add_node(
        "empty_response",
        empty_response
    )

    graph_builder.add_node(
        "error_response",
        error_response
    )

    graph_builder.add_node(
        "loop_limit_response",
        loop_limit_response
    )

    # ==================================================
    # GRAPH EDGES
    # ==================================================

    graph_builder.add_edge(
        START,
        "agent"
    )

    # Agent dynamically decides:
    #   tool call -> tools
    #   no tool   -> END
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition
    )

    # Tool result is checked by guardrails.
    graph_builder.add_conditional_edges(
        "tools",
        guardrail_check,
        {
            "empty": "empty_response",
            "error": "error_response",
            "valid": "final_answer",
            "loop_limit": "loop_limit_response",
        }
    )

    graph_builder.add_edge(
        "final_answer",
        END
    )

    graph_builder.add_edge(
        "empty_response",
        END
    )

    graph_builder.add_edge(
        "error_response",
        END
    )

    graph_builder.add_edge(
        "loop_limit_response",
        END
    )

    graph = graph_builder.compile()

    # ==================================================
    # 12. EXECUTE GRAPH
    # ==================================================

    print("\n" + "=" * 70)
    print("LANGGRAPH MCP AGENT")
    print("=" * 70)
    print(f"USER QUESTION: {question}")
    print("=" * 70)

    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    # ==================================================
    # 13. FINAL ANSWER
    # ==================================================

    final_message = result["messages"][-1]
    final_answer = final_message.content

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final_answer)

    # ==================================================
    # 14. OBSERVABILITY / MANUAL EVALUATION
    # ==================================================

    cache_stats = tool_cache.stats()

    cost_summary = cost_calculator.calculate(
        total_input_tokens,
        total_output_tokens
    )

    trace.finish(status="success")

    total_latency = (
        time.perf_counter() - total_start
    )

    print("\n" + "=" * 70)
    print("AGENT EXECUTION SUMMARY")
    print("=" * 70)

    print(f"\nUser Question:")
    print(question)

    print("\nTool Used:")

    if selected_tool_calls:
        print("YES")

        for index, tool_call in enumerate(
            selected_tool_calls,
            start=1
        ):
            print(f"\nTool Call #{index}")
            print(f"Tool      : {tool_call['tool']}")
            print(f"Arguments : {tool_call['arguments']}")

    else:
        print("NO")
        print("Answer source: Agent's own knowledge")

    print("\nTool Result:")

    if executed_tool_results:
        for index, tool_result in enumerate(
            executed_tool_results,
            start=1
        ):
            print(f"\nResult #{index}")
            print(f"Tool   : {tool_result['tool']}")
            print(f"Source : {tool_result['source']}")

            result_value = tool_result["result"]

            if isinstance(result_value, (dict, list)):
                print(
                    json.dumps(
                        result_value,
                        indent=2,
                        ensure_ascii=False,
                        default=str
                    )
                )
            else:
                print(result_value)
    else:
        print("No tool result because no MCP tool was used.")

    print("\nAnswer Source:")

    if executed_tool_results:
        print("MCP TOOL RESULT -> GROUNDED LLM ANSWER")
    else:
        print("LLM KNOWLEDGE - generated by Qwen3:4b")

    print("\nManual Correctness Check:")
    print(
        "CHECK THE USER QUESTION, SELECTED TOOL, "
        "TOOL RESULT, AND FINAL ANSWER ABOVE."
    )

    print("\nTool Calls        :", tool_call_count)
    print("Input Tokens      :", total_input_tokens)
    print("Output Tokens     :", total_output_tokens)
    print(
        "Total Tokens      :",
        total_input_tokens + total_output_tokens
    )
    print(
        "Cache Statistics  :",
        cache_stats
    )
    print(
        "Total Cost        :",
        f"${cost_summary['total_cost']:.6f}"
    )
    print(
        "Total Latency     :",
        f"{total_latency:.2f} sec"
    )
    print(
        "Trace ID          :",
        trace.trace_id
    )

    print("=" * 70)

    logger.info(
        "Agent execution completed | Total Latency=%.2f sec",
        total_latency
    )

    return {
        "question": question,
        "selected_tool": (
            selected_tool_calls[0]["tool"]
            if len(selected_tool_calls) == 1
            else [item["tool"] for item in selected_tool_calls]
            if selected_tool_calls
            else None
        ),
        "tool_arguments": (
            selected_tool_calls[0]["arguments"]
            if len(selected_tool_calls) == 1
            else [item["arguments"] for item in selected_tool_calls]
            if selected_tool_calls
            else {}
        ),
        "selected_tools": selected_tool_calls,
        "tool_result": (
            executed_tool_results[0]["result"]
            if len(executed_tool_results) == 1
            else [item["result"] for item in executed_tool_results]
            if executed_tool_results
            else None
        ),
        "tool_results": executed_tool_results,
        "final_answer": final_answer,
        "answer_source": (
            "tool"
            if executed_tool_results
            else "knowledge"
        ),
        "tool_calls": tool_call_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "cache_stats": cache_stats,
        "cost_summary": cost_summary,
        "total_latency": total_latency,
        "trace_id": trace.trace_id,
    }


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    question = input("\nEnter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
    else:
        asyncio.run(main(question))
