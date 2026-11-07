from typing import Literal, Annotated
import operator
from urllib import response
from pydantic import BaseModel
import asyncio
import json
import re
import traceback

from langchain_ollama import ChatOllama
from multi_agent.rag_agent import rag_agent as run_rag_agent
from multi_agent.routing import route_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient


# ============================================================
# MODEL
# ============================================================

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    num_predict=1024,
    reasoning=False
)


# ============================================================
# MCP CLIENT
# ============================================================

client = MultiServerMCPClient(
    {
        "weather": {
            "command": "python",
            "args": ["mcp_server/weather_server.py"],
            "transport": "stdio",
        },
        "database": {
            "command": "python",
            "args": ["mcp_server/db_server.py"],
            "transport": "stdio",
        },
    }
)


# ============================================================
# STATE
# ============================================================

class AgentState(MessagesState):
    user_question: str
    next_agent: str
    completed_agents: Annotated[list[str], operator.add]


class RouteDecision(BaseModel):
    next_agent: Literal[
        "employee_agent",
        "weather_agent",
        "rag_agent",
        "FINISH"
    ]


# ============================================================
# SUPERVISOR
# ============================================================


def supervisor(state: AgentState):

    question = state["user_question"]
    completed_agents = state.get("completed_agents", [])

    router_llm = llm.with_structured_output(RouteDecision)

    prompt = f"""
You are the supervisor of a multi-agent system.

Your job is to decide which agent should execute NEXT.

Available agents:

employee_agent:
- Employee information
- Employee ID
- Employee name
- Salary
- Department
- Employee database questions

weather_agent:
- Current weather
- Temperature
- Humidity
- Wind
- Weather forecast

rag_agent:
- Questions about documents
- Climate documents
- Evidence from the knowledge base
- Information contained in the RAG documents

You can also choose FINISH when all required tasks
have been completed.

IMPORTANT RULES:

1. Identify ALL tasks required by the original question.
2. Do NOT select an agent that has already completed its task.
3. If multiple agents are required, select them ONE AT A TIME.
4. Select FINISH when no required tasks remain.
5. Never select an agent for an unrelated task.
6. Never invent information.

Original user question:
{question}

Agents already completed:
{completed_agents}

Choose exactly ONE:
employee_agent
weather_agent
rag_agent
FINISH
"""

    decision = router_llm.invoke([
        SystemMessage(content=prompt)
    ])

    selected_agent = decision.next_agent

    print(f"DEBUG question: {question}")
    print(f"DEBUG completed_agents: {completed_agents}")
    print(f"DEBUG LLM decision: {selected_agent}")

    # --------------------------------------------------------
    # SAFETY CHECK 1: Do not execute completed agents again.
    # Ask the router again is not necessary: FINISH is safe
    # only when no required tasks remain, which this check
    # alone cannot establish.
    # --------------------------------------------------------

    if selected_agent in completed_agents:
        selected_agent = "FINISH"

    # --------------------------------------------------------
    # SAFETY CHECK 2: Prevent weather routing when the
    # question clearly does not ask about weather.
    # --------------------------------------------------------

    question_lower = question.lower()

    weather_keywords = [
        "weather",
        "forecast",
        "humidity",
        "wind speed",
        "wind direction",
        "raining",
        "rain",
        "sunny",
        "cloudy",
        "current conditions",
    ]

    asks_weather = any(
        keyword in question_lower
        for keyword in weather_keywords
    )

    if (
        selected_agent == "weather_agent"
        and not asks_weather
        and "weather_agent" not in completed_agents
    ):
        print(
            "DEBUG: Rejected weather_agent because "
            "the question does not request weather."
        )

        # Let the router select the appropriate remaining task.
        retry_prompt = f"""
The original question is:
{question}

Completed agents:
{completed_agents}

Your previous decision selected weather_agent, but the
question does not request weather information.

Choose the next agent that is actually required:
employee_agent, rag_agent, or FINISH.

Do not select completed agents.
"""

        retry_decision = router_llm.invoke([
            SystemMessage(content=retry_prompt)
        ])

        selected_agent = retry_decision.next_agent

        # Final safeguard against routing to a completed agent.
        if selected_agent in completed_agents:
            selected_agent = "FINISH"

    print(f"\nSupervisor selected: {selected_agent}")

    return {
        "next_agent": selected_agent
    }


# ============================================================
# EMPLOYEE AGENT
# ============================================================

async def employee_agent(state: AgentState):

    question = state["user_question"]

    # Get MCP tools
    tools = await client.get_tools()

    employee_tools = [
        tool
        for tool in tools
        if tool.name in [
            "get_employee",
            "get_employees_by_department",
            "search_employee_by_name"
        ]
    ]

    # Let LLM select the appropriate employee tool
    employee_llm = llm.bind_tools(employee_tools)

    response = await employee_llm.ainvoke([
        SystemMessage(
            content="""
You are the Employee Agent.

Use the available employee database tools to answer
employee-related questions.

Always use the appropriate database tool when the
user asks for employee information.

Do not invent employee information.

You are responsible only for employee database queries.

Ignore requests about weather, climate, documents, or other
agents. The Supervisor handles those tasks.

When the user asks for employees in a department, immediately
call get_employees_by_department with the department name.

Do not explain your reasoning. Do not answer from memory.
Use the database tool to retrieve employee information.
"""
        ),
        HumanMessage(content=question)
    ])
    print(f"DEBUG employee response: {response}")
    print(f"DEBUG employee tool calls: {response.tool_calls}")

    # Check whether LLM requested a tool
    if response.tool_calls:

        tool_call = response.tool_calls[0]

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"\nEmployee Agent selected tool: {tool_name}")
        print(f"Arguments: {tool_args}")

        # Find selected MCP tool
        selected_tool = next(
            tool
            for tool in employee_tools
            if tool.name == tool_name
        )

        # Execute MCP tool
        tool_result = await selected_tool.ainvoke(tool_args)

        print(f"\nMCP Tool Result: {tool_result}")

        # -----------------------------------------
        # Process ALL MCP result items
        # -----------------------------------------

        all_results = []

        for item in tool_result:

            if isinstance(item, dict) and "text" in item:

                text = item["text"]

                try:
                    data = json.loads(text)
                    all_results.append(data)

                except json.JSONDecodeError:
                    all_results.append(text)

        # -----------------------------------------
        # Handle error
        # -----------------------------------------

        if len(all_results) == 1 and isinstance(all_results[0], dict):

            if "error" in all_results[0]:

                return {
                    "messages": [
                        HumanMessage(
                            content=all_results[0]["error"]
                        )
                    ],
                    "completed_agents": ["employee_agent"]
                }

        # -----------------------------------------
        # Single employee result
        # -----------------------------------------

        if len(all_results) == 1 and isinstance(all_results[0], dict):

            employee = all_results[0]

            answer = (
                f"Employee ID: {employee.get('employee_id')}\n"
                f"Name: {employee.get('name')}\n"
                f"Department: {employee.get('department')}\n"
                f"Designation: {employee.get('designation')}\n"
                f"Salary: {employee.get('salary')}"
            )

            return {
                "messages": [
                    HumanMessage(content=answer)
                ],
                "completed_agents": ["employee_agent"]
            }

        # -----------------------------------------
        # Multiple employee results
        # -----------------------------------------

        employees = []

        for item in all_results:

            if isinstance(item, dict):

                # If MCP returns a list inside one item
                if isinstance(item.get("employees"), list):

                    employees.extend(item["employees"])

                # If item itself is an employee
                elif "employee_id" in item:

                    employees.append(item)

            elif isinstance(item, list):

                employees.extend(item)

        if employees:

            lines = ["Employees found:"]

            for employee in employees:

                lines.append(
                    f"- {employee.get('employee_id')}: "
                    f"{employee.get('name')} | "
                    f"{employee.get('department')} | "
                    f"{employee.get('designation')} | "
                    f"Salary: {employee.get('salary')}"
                )

            return {
                "messages": [
                    HumanMessage(
                        content="\n".join(lines)
                    )
                ],
                "completed_agents": ["employee_agent"]
            }

        return {
            "messages": [
                HumanMessage(
                    content="No employees were found."
                )
            ],
            "completed_agents": ["employee_agent"]
        }

    # No tool selected
    
    return {
        "messages": [
            HumanMessage(
                content=(
                    "Employee Agent could not execute the request "
                    "because the model did not select a database tool."
                )
            )
        ],
        "completed_agents": ["employee_agent"]
    }


# ============================================================
# WEATHER AGENT
# ============================================================

async def weather_agent(state: AgentState):

    # question = state["messages"][-1].content.strip()
    question = state["user_question"].strip()

    # --------------------------------------------------------
    # Get MCP tools
    # --------------------------------------------------------
    print("DEBUG: calling client.get_tools()")
    tools = await client.get_tools()
    print("DEBUG: client.get_tools() completed")
    print(f"DEBUG: available tools: {[tool.name for tool in tools]}")

    weather_tool = next(
        tool
        for tool in tools
        if tool.name == "get_current_weather"
    )

    # --------------------------------------------------------
    # Extract city directly from the user question
    # --------------------------------------------------------

    
    
    city_match = re.search(
        r"\b(?:in|of)\s+([A-Za-z][A-Za-z\s]*?)(?=\s+and\b|[?!.,]|$)",
        question,
        re.IGNORECASE
    )


    print(f"DEBUG: question before city extraction: {question!r}")

    if city_match:
        city = city_match.group(1).strip().title()
        print(f"DEBUG extracted city: {city}")
    else:
        return {
            "messages": [
                HumanMessage(
                    content="Could not identify the city from the question."
                )
            ]
        }

    print("\nWeather Agent selected tool: get_current_weather")
    print(f"Arguments: {{'city': '{city}'}}")

    # --------------------------------------------------------
    # Execute MCP weather tool
    # --------------------------------------------------------

    tool_result = await weather_tool.ainvoke({
        "city": city
    })

    print(f"\nMCP Tool Result: {tool_result}")

    # --------------------------------------------------------
    # Extract tool response
    # --------------------------------------------------------

    tool_text = tool_result[0]["text"]

    # --------------------------------------------------------
    # Handle non-JSON errors
    # --------------------------------------------------------

    if not tool_text.strip().startswith("{"):

        return {
            "messages": [
                HumanMessage(
                    content=f"Weather tool failed: {tool_text}"
                )
            ]
        }

    # --------------------------------------------------------
    # Convert JSON string to dictionary
    # --------------------------------------------------------

    weather_data = json.loads(tool_text)

    # --------------------------------------------------------
    # Handle API error
    # --------------------------------------------------------

    if "error" in weather_data:

        return {
            "messages": [
                HumanMessage(
                    content=weather_data["error"]
                )
            ]
        }

    # --------------------------------------------------------
    # Extract current weather
    # --------------------------------------------------------

    current = weather_data["current_weather"]

    # --------------------------------------------------------
    # Build final answer
    # --------------------------------------------------------

    answer = (
        f"Current weather in {weather_data['city']}, "
        f"{weather_data['country']}:\n"
        f"Temperature: {current['temperature_2m']}°C\n"
        f"Humidity: {current['relative_humidity_2m']}%\n"
        f"Wind speed: {current['wind_speed_10m']} m/s"
    )

    return {
        "messages": [
            HumanMessage(content=answer)
        ],
        "completed_agents": ["weather_agent"]
    }

async def rag_agent_node(state: AgentState):

    question = state["user_question"].strip()

    print("\nRAG Agent selected")

    answer = await run_rag_agent(question)

    return {
        "messages": [
            HumanMessage(content=answer)
        ],
        "completed_agents": ["rag_agent"]
    }


# ============================================================
# ROUTER
# ============================================================

# def route_agent(
#     state: AgentState
# ) -> Literal[
#     "employee_agent",
#     "weather_agent",
#     "rag_agent",
#     "FINISH"
# ]:
#     return state["next_agent"]

# ============================================================
# BUILD GRAPH
# ============================================================

graph = StateGraph(AgentState)

graph.add_node(
    "supervisor",
    supervisor
)

graph.add_node(
    "employee_agent",
    employee_agent
)

graph.add_node(
    "weather_agent",
    weather_agent
)

graph.add_node(
    "rag_agent",
    rag_agent_node
)

# START -> SUPERVISOR
graph.add_edge(
    START,
    "supervisor"
)

# SUPERVISOR -> SELECTED AGENT
graph.add_conditional_edges(
    "supervisor",
    route_agent,
    {
        "employee_agent": "employee_agent",
        "weather_agent": "weather_agent",
        "rag_agent": "rag_agent",
        "FINISH": END
    }
)

# AGENTS -> SUPERVISOR
graph.add_edge(
    "employee_agent",
    "supervisor"
)

graph.add_edge(
    "weather_agent",
    "supervisor"
)

graph.add_edge(
    "rag_agent",
    "supervisor"
)


#Compile graph
app = graph.compile()


# ============================================================
# RUN APPLICATION
# ============================================================

async def main():

    print("\n========================================")
    print("       MULTI-AGENT SYSTEM")
    print("========================================")
    print("Enter your questions below.")
    print("Press Ctrl+C to exit.")
    print("========================================")

    while True:

        try:
            question = input("\nEnter your question: ").strip()

        except EOFError:
            break

        if not question:
            continue
        print(f"DEBUG input question: {question}")
        try:

            result = await app.ainvoke({
                "messages": [
                    HumanMessage(content=question)
                ],
                "user_question": question,
                "completed_agents": []
            })

            
            print("\n==============================")
            print("FINAL ANSWER")
            print("==============================")

            # The first message is the original user question.
            # All subsequent messages contain agent responses.
            agent_messages = result.get("messages", [])[1:]

            if agent_messages:
                for index, message in enumerate(agent_messages, start=1):
                    print(f"\n--- Agent Result {index} ---")
                    print(message.content)
            else:
                print("No agent results were returned.")


        except Exception as e:
            traceback.print_exc()
            print("\n==============================")
            print("ERROR")
            print("==============================")

            print(f"{type(e).__name__}: {e}")

            print("\nThe Multi-Agent System is still running.")


if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:

        print("\n\nMulti-Agent System stopped.")