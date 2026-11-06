import asyncio
import json
import re

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from langchain_core.messages import ToolMessage

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# 1. EVALUATION DATASET
# ============================================================

evaluation_dataset = [

    {
        "question": "Find the employee with ID EMP001.",

        "expected_tool": "get_employee",

        "expected_tool_args": {
            "employee_id": "EMP001"
        },

        "expected_employee_data": {
            "employee_id": "EMP001",
            "name": "Ravi Kumar",
            "department": "Engineering",
            "designation": "Software Engineer",
            "salary": 75000.0
        }
    },

    {
        "question": "Find the employee with ID EMP003.",

        "expected_tool": "get_employee",

        "expected_tool_args": {
            "employee_id": "EMP003"
        },

        "expected_employee_data": {
            "employee_id": "EMP003",
            "name": "Rahul Reddy",
            "department": "Engineering",
            "designation": "Data Scientist",
            "salary": 95000.0
        }
    },

    {
        "question": "Find the employee with ID EMP005.",

        "expected_tool": "get_employee",

        "expected_tool_args": {
            "employee_id": "EMP005"
        },

        "expected_employee_data": {
            "employee_id": "EMP005",
            "name": "Arjun Patel",
            "department": "Engineering",
            "designation": "AI Engineer",
            "salary": 110000.0
        }
    }
]


# ============================================================
# 2. BUILD AGENT
# ============================================================

async def build_agent():

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

    tools = await client.get_tools()

    print(f"MCP tools loaded: {len(tools)}")

    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
    )

    llm_with_tools = llm.bind_tools(tools)

    captured_tool_calls = []

    async def agent(state: MessagesState):

        response = await llm_with_tools.ainvoke(
            state["messages"]
        )

        if response.tool_calls:

            for tool_call in response.tool_calls:

                captured_tool_calls.append(
                    {
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                    }
                )

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

    graph_builder = StateGraph(MessagesState)

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

    return graph, captured_tool_calls


# ============================================================
# 3. EVALUATE TOOL SELECTION
# ============================================================

def evaluate_tool_selection(
    captured_tool_calls,
    expected_tool
):

    if not captured_tool_calls:
        return False

    actual_tool = captured_tool_calls[0]["name"]

    return actual_tool == expected_tool


# ============================================================
# 4. EVALUATE TOOL ARGUMENTS
# ============================================================

def evaluate_tool_arguments(
    captured_tool_calls,
    expected_tool_args
):

    if not captured_tool_calls:
        return False

    actual_args = captured_tool_calls[0]["args"]

    return actual_args == expected_tool_args


# ============================================================
# 5. EXTRACT TOOL RESULT
# ============================================================

def extract_tool_result(result):

    for message in result["messages"]:

        if isinstance(message, ToolMessage):

            content = message.content

            if isinstance(content, list):

                text_parts = []

                for item in content:

                    if isinstance(item, dict):

                        if "text" in item:
                            text_parts.append(
                                item["text"]
                            )

                    else:
                        text_parts.append(
                            str(item)
                        )

                content = "".join(text_parts)

            try:
                return json.loads(content)

            except (json.JSONDecodeError, TypeError):
                return None

    return None


# ============================================================
# 6. EVALUATE TOOL EXECUTION
# ============================================================

def evaluate_tool_execution(
    tool_result
):

    if tool_result is None:
        return False

    if isinstance(tool_result, dict):

        if "error" in tool_result:
            return False

    if tool_result == []:
        return False

    return True


# ============================================================
# 7. EVALUATE EMPLOYEE DATA
# ============================================================

def evaluate_employee_data(
    actual_data,
    expected_data
):

    field_results = {}

    # --------------------------------------------------------
    # Handle MCP result format
    # --------------------------------------------------------

    if isinstance(actual_data, list):

        if len(actual_data) == 0:
            return {
                field: False
                for field in expected_data
            }

        actual_data = actual_data[0]

    # --------------------------------------------------------
    # Compare every field
    # --------------------------------------------------------

    for field, expected_value in expected_data.items():

        actual_value = actual_data.get(field)

        field_results[field] = (
            actual_value == expected_value
        )

    return field_results

def evaluate_salary(expected_salary, final_answer):

    expected_salary = float(expected_salary)

    # Find numbers that look like salary values
    matches = re.findall(
        r'\$?\s?[\d,]+(?:\.\d+)?',
        final_answer
    )

    for value in matches:

        cleaned_value = (
            value
            .replace("$", "")
            .replace(",", "")
            .strip()
        )

        try:

            actual_salary = float(
                cleaned_value
            )

            if actual_salary == expected_salary:
                return True

        except ValueError:
            continue

    return False
# ============================================================
# 8. EVALUATE FINAL ANSWER
# ============================================================

def evaluate_final_answer(
    result,
    expected_data
):

    final_message = result["messages"][-1]

    final_answer = final_message.content

    if not isinstance(final_answer, str):
        final_answer = str(final_answer)

    final_answer_lower = final_answer.lower()

    # Check important expected values
    checks = {

        "employee_id":
            str(
                expected_data["employee_id"]
            ).lower()
            in final_answer_lower,

        "name":
            expected_data["name"].lower()
            in final_answer_lower,

        "department":
            expected_data["department"].lower()
            in final_answer_lower,

        "designation":
            expected_data["designation"].lower()
            in final_answer_lower,

        "salary":
            evaluate_salary(
                expected_data["salary"],
                final_answer
            ),
    }

    return checks


# ============================================================
# 9. RUN ONE TEST CASE
# ============================================================

async def run_test_case(
    graph,
    captured_tool_calls,
    test_case
):

    captured_tool_calls.clear()

    question = test_case["question"]

    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    # --------------------------------------------------------
    # Tool selection
    # --------------------------------------------------------

    tool_selection_pass = evaluate_tool_selection(
        captured_tool_calls,
        test_case["expected_tool"]
    )

    # --------------------------------------------------------
    # Tool arguments
    # --------------------------------------------------------

    tool_argument_pass = evaluate_tool_arguments(
        captured_tool_calls,
        test_case["expected_tool_args"]
    )

    # --------------------------------------------------------
    # Extract actual tool result
    # --------------------------------------------------------

    actual_tool_result = extract_tool_result(
        result
    )

    # --------------------------------------------------------
    # Tool execution
    # --------------------------------------------------------

    tool_execution_pass = evaluate_tool_execution(
        actual_tool_result
    )

    # --------------------------------------------------------
    # Employee data
    # --------------------------------------------------------

    employee_field_results = evaluate_employee_data(
        actual_tool_result,
        test_case["expected_employee_data"]
    )

    employee_data_pass = all(
        employee_field_results.values()
    )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    final_answer_results = evaluate_final_answer(
        result,
        test_case["expected_employee_data"]
    )

    final_answer_pass = all(
        final_answer_results.values()
    )

    # --------------------------------------------------------
    # Overall task success
    # --------------------------------------------------------

    task_success = (
        tool_selection_pass
        and tool_argument_pass
        and tool_execution_pass
        and employee_data_pass
        and final_answer_pass
    )

    return {

        "question": question,

        "tool_selection":
            tool_selection_pass,

        "tool_arguments":
            tool_argument_pass,

        "tool_execution":
            tool_execution_pass,

        "employee_data":
            employee_data_pass,

        "employee_fields":
            employee_field_results,

        "final_answer":
            final_answer_pass,

        "final_answer_fields":
            final_answer_results,

        "task_success":
            task_success,

        "tool_calls":
            captured_tool_calls.copy(),

        "tool_result":
            actual_tool_result,

        "answer":
            result["messages"][-1].content,
    }


# ============================================================
# 10. PRINT TEST RESULT
# ============================================================

def print_test_result(
    test_number,
    result
):

    print("\n" + "=" * 60)

    print(
        f"TEST CASE {test_number}"
    )

    print("=" * 60)

    print(
        f"\nQuestion: {result['question']}"
    )

    print(
        "\nTool Selection:   "
        f"{'PASS' if result['tool_selection'] else 'FAIL'}"
    )

    print(
        "Tool Arguments:   "
        f"{'PASS' if result['tool_arguments'] else 'FAIL'}"
    )

    print(
        "Tool Execution:   "
        f"{'PASS' if result['tool_execution'] else 'FAIL'}"
    )

    print(
        "\nEmployee Data:"
    )

    for field, passed in result["employee_fields"].items():

        print(
            f"  {field:<15} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print(
        "\nFinal Answer:"
    )

    for field, passed in result["final_answer_fields"].items():

        print(
            f"  {field:<15} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print(
        "\nTask Success:     "
        f"{'PASS' if result['task_success'] else 'FAIL'}"
    )

    print(
        "\nTool Calls:"
    )

    print(
        json.dumps(
            result["tool_calls"],
            indent=2
        )
    )

    print(
        "\nTool Result:"
    )

    print(
        json.dumps(
            result["tool_result"],
            indent=2
        )
    )

    print(
        "\nAgent Answer:"
    )

    print(
        result["answer"]
    )


# ============================================================
# 11. PRINT SUMMARY
# ============================================================

def print_summary(results):

    total_tests = len(results)

    tool_selection_passes = sum(
        result["tool_selection"]
        for result in results
    )

    tool_argument_passes = sum(
        result["tool_arguments"]
        for result in results
    )

    tool_execution_passes = sum(
        result["tool_execution"]
        for result in results
    )

    employee_data_passes = sum(
        result["employee_data"]
        for result in results
    )

    final_answer_passes = sum(
        result["final_answer"]
        for result in results
    )

    task_successes = sum(
        result["task_success"]
        for result in results
    )

    print(
        "\n\n" + "=" * 60
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"\nTotal Tests: "
        f"{total_tests}"
    )

    print(
        f"Tool Selection Accuracy: "
        f"{tool_selection_passes / total_tests * 100:.2f}%"
    )

    print(
        f"Tool Argument Accuracy: "
        f"{tool_argument_passes / total_tests * 100:.2f}%"
    )

    print(
        f"Tool Execution Success: "
        f"{tool_execution_passes / total_tests * 100:.2f}%"
    )

    print(
        f"Employee Data Accuracy: "
        f"{employee_data_passes / total_tests * 100:.2f}%"
    )

    print(
        f"Final Answer Accuracy: "
        f"{final_answer_passes / total_tests * 100:.2f}%"
    )

    print(
        f"Task Success Rate: "
        f"{task_successes / total_tests * 100:.2f}%"
    )

    print(
        "\n" + "=" * 60
    )


# ============================================================
# 12. MAIN
# ============================================================

async def main():

    print(
        "\n" + "=" * 60
    )

    print(
        "AGENT EVALUATION"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Build agent once
    # --------------------------------------------------------

    graph, captured_tool_calls = await build_agent()

    results = []

    # --------------------------------------------------------
    # Execute test cases
    # --------------------------------------------------------

    for index, test_case in enumerate(
        evaluation_dataset,
        start=1
    ):

        result = await run_test_case(
            graph,
            captured_tool_calls,
            test_case
        )

        results.append(result)

        print_test_result(
            index,
            result
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        results
    )


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )