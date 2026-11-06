import asyncio
import re

from langgraph_mcp_agent import main


# ============================================================
# EVALUATION TEST CASES
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # 1. DATABASE - EMPLOYEE BY ID
    # --------------------------------------------------------

    {
        "question": "Find the employee with ID EMP001.",
        "expected_tool": "get_employee",
        "expected_arguments": {
            "employee_id": "EMP001"
        },
        "expected_values": [
            "EMP001",
            "Ravi Kumar",
            "Engineering",
            "Software Engineer",
            "75000",
        ],
    },

    # --------------------------------------------------------
    # 2. DATABASE - EMPLOYEES BY DEPARTMENT
    # --------------------------------------------------------

    {
        "question": "Show me all employees in the Engineering department.",
        "expected_tool": "get_employees_by_department",
        "expected_arguments": {
            "department": "Engineering"
        },
        "expected_values": [
            "Ravi Kumar",
            "Rahul Reddy",
            "Arjun Patel",
        ],
    },

    # --------------------------------------------------------
    # 3. DATABASE - SEARCH EMPLOYEE BY NAME
    # --------------------------------------------------------

    {
        "question": "Find the employee named Rahul.",
        "expected_tool": "search_employee_by_name",
        "expected_arguments": {
            "name": "Rahul"
        },
        "expected_values": [
            "Rahul Reddy",
            "Engineering",
            "Data Scientist",
            "95000",
        ],
    },

    # --------------------------------------------------------
    # 4. WEATHER
    # --------------------------------------------------------

    {
        "question": "What is the current weather in Anantapur?",
        "expected_tool": "get_current_weather",
        "expected_arguments": {
            "city": "Anantapur"
        },
        "expected_values": [
            "Anantapur",
        ],
    },

    # --------------------------------------------------------
    # 5. DOCUMENT - COMPANY POLICY
    # --------------------------------------------------------

    {
        "question": "Show me the company policy.",
        "expected_tool": "get_company_policy",
        "expected_arguments": {},
        "expected_values": [
            "company_policy",
        ],
    },

    # --------------------------------------------------------
    # 6. DOCUMENT - CSV EMPLOYEE SEARCH
    # --------------------------------------------------------

    {
        "question": "Find the employee named John in the employee CSV.",
        "expected_tool": "search_csv_employee",
        "expected_arguments": {
            "name": "John"
        },
        "expected_values": [
            "John Smith",
            "EMP101",
            "Engineering",
            "Software Engineer",
        ],
    },

    # --------------------------------------------------------
    # 7. DOCUMENT - CSV DEPARTMENT SEARCH
    # --------------------------------------------------------

    {
        "question": "Show me all employees in the Engineering department from the CSV.",
        "expected_tool": "get_csv_employees_by_department",
        "expected_arguments": {
            "department": "Engineering"
        },
        "expected_values": [
            "John Smith",
            "Mike Davis",
            "David Wilson",
        ],
    },

    # --------------------------------------------------------
    # 8. DOCUMENT - LIST FILES
    # --------------------------------------------------------

    {
        "question": "List all available documents.",
        "expected_tool": "list_documents",
        "expected_arguments": {},
        "expected_values": [
            "company_policy.txt",
            "employees.csv",
        ],
    },

    # --------------------------------------------------------
    # 9. CALCULATOR - ADD
    # --------------------------------------------------------

    {
        "question": "Calculate 25 plus 15.",
        "expected_tool": "add",
        "expected_arguments": {
            "a": 25,
            "b": 15,
        },
        "expected_values": [
            "40",
        ],
    },

    # --------------------------------------------------------
    # 10. CALCULATOR - MULTIPLY
    # --------------------------------------------------------

    {
        "question": "Calculate 12 multiplied by 8.",
        "expected_tool": "multiply",
        "expected_arguments": {
            "a": 12,
            "b": 8,
        },
        "expected_values": [
            "96",
        ],
    },

    # --------------------------------------------------------
    # 11. CALCULATOR - DIVIDE
    # --------------------------------------------------------

    {
        "question": "Calculate 100 divided by 4.",
        "expected_tool": "divide",
        "expected_arguments": {
            "a": 100,
            "b": 4,
        },
        "expected_values": [
            "25",
        ],
    },
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value):
    """
    Normalize text for reliable comparison.
    """

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower(),
    )


def normalize_numeric(value):
    """
    Normalize numeric values.

    Examples:

    110000
    110,000
    110000.0
    110,000.0

    all become:

    110000
    """

    text = str(value).strip()

    text = text.replace(",", "")

    try:
        number = float(text)

        if number.is_integer():
            return str(int(number))

        return str(number)

    except ValueError:

        return text.lower()


# ============================================================
# ARGUMENT EVALUATION
# ============================================================

def evaluate_arguments(
    expected_arguments,
    actual_arguments,
):
    """
    Compare expected tool arguments with
    arguments generated by the agent.
    """

    if not expected_arguments:
        return True

    if not actual_arguments:
        return False

    for key, expected_value in expected_arguments.items():

        if key not in actual_arguments:
            return False

        actual_value = actual_arguments[key]

        # Numeric comparison
        if isinstance(expected_value, (int, float)):

            expected_normalized = normalize_numeric(
                expected_value
            )

            actual_normalized = normalize_numeric(
                actual_value
            )

            if expected_normalized != actual_normalized:
                return False

        # String comparison
        else:

            expected_normalized = normalize(
                expected_value
            )

            actual_normalized = normalize(
                actual_value
            )

            if expected_normalized != actual_normalized:
                return False

    return True


# ============================================================
# TOOL RESULT EVALUATION
# ============================================================

def evaluate_tool_result(
    expected_values,
    tool_result,
):
    """
    Verify that expected values exist
    in the MCP tool result.
    """

    result_text = normalize(tool_result)

    for expected_value in expected_values:

        # Numeric value
        if isinstance(expected_value, (int, float)):

            expected_normalized = normalize_numeric(
                expected_value
            )

            result_normalized = normalize_numeric(
                tool_result
            )

            if expected_normalized not in result_normalized:
                return False

        else:

            expected_normalized = normalize(
                expected_value
            )

            if expected_normalized not in result_text:
                return False

    return True


# ============================================================
# FINAL ANSWER EVALUATION
# ============================================================

def evaluate_final_answer(
    expected_values,
    final_answer,
):
    """
    Verify that important expected information
    appears in the final answer.
    """

    answer_text = normalize(final_answer)

    for expected_value in expected_values:

        if isinstance(expected_value, (int, float)):

            expected_normalized = normalize_numeric(
                expected_value
            )

            answer_normalized = normalize_numeric(
                final_answer
            )

            if expected_normalized not in answer_normalized:
                return False

        else:

            expected_normalized = normalize(
                expected_value
            )

            if expected_normalized not in answer_text:
                return False

    return True


# ============================================================
# COMPLETE TEST CASE EVALUATION
# ============================================================

def evaluate_test_case(
    test_case,
    agent_result,
):
    """
    Evaluate one complete agent execution.
    """

    actual_tool = agent_result.get(
        "selected_tool"
    )

    actual_arguments = agent_result.get(
        "tool_arguments",
        {},
    )

    tool_result = agent_result.get(
        "tool_result",
        "",
    )

    final_answer = agent_result.get(
        "final_answer",
        "",
    )

    # --------------------------------------------------------
    # Tool selection
    # --------------------------------------------------------

    tool_pass = (
        actual_tool
        == test_case["expected_tool"]
    )

    # --------------------------------------------------------
    # Tool arguments
    # --------------------------------------------------------

    arguments_pass = evaluate_arguments(
        test_case["expected_arguments"],
        actual_arguments,
    )

    # --------------------------------------------------------
    # MCP tool result
    # --------------------------------------------------------

    tool_data_pass = evaluate_tool_result(
        test_case["expected_values"],
        tool_result,
    )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    final_answer_pass = evaluate_final_answer(
        test_case["expected_values"],
        final_answer,
    )

    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    overall_pass = (
        tool_pass
        and arguments_pass
        and tool_data_pass
        and final_answer_pass
    )

    return {
        "tool": tool_pass,
        "arguments": arguments_pass,
        "tool_data": tool_data_pass,
        "final_answer": final_answer_pass,
        "overall": overall_pass,
        "actual_tool": actual_tool,
        "actual_arguments": actual_arguments,
    }


# ============================================================
# PRINT INDIVIDUAL RESULT
# ============================================================

def print_test_result(
    test_number,
    test_case,
    result,
):
    """

    Print evaluation result for one test.
    """

    print()

    print("=" * 75)

    print(
        f"TEST CASE {test_number}"
    )

    print("=" * 75)

    print(
        f"Question: "
        f"{test_case['question']}"
    )

    print()

    print(
        f"Expected Tool : "
        f"{test_case['expected_tool']}"
    )

    print(
        f"Actual Tool   : "
        f"{result['actual_tool']}"
    )

    print()

    print(
        f"Tool Selection : "
        f"{'PASS' if result['tool'] else 'FAIL'}"
    )

    print(
        f"Arguments      : "
        f"{'PASS' if result['arguments'] else 'FAIL'}"
    )

    print(
        f"Tool Data      : "
        f"{'PASS' if result['tool_data'] else 'FAIL'}"
    )

    print(
        f"Final Answer   : "
        f"{'PASS' if result['final_answer'] else 'FAIL'}"
    )

    print(
        f"Overall        : "
        f"{'PASS' if result['overall'] else 'FAIL'}"
    )


# ============================================================
# RUN COMPLETE EVALUATION
# ============================================================

async def run_evaluation():

    print()

    print("=" * 75)

    print(
        "LANGGRAPH MCP AGENT EVALUATION"
    )

    print("=" * 75)

    print(
        f"Total test cases: "
        f"{len(TEST_CASES)}"
    )

    results = []

    # --------------------------------------------------------
    # Run every test
    # --------------------------------------------------------

    for index, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()

        print(
            f"Running test "
            f"{index}/{len(TEST_CASES)}..."
        )

        try:

            agent_result = await main(
                test_case["question"]
            )

            evaluation_result = evaluate_test_case(
                test_case,
                agent_result,
            )

            results.append(
                evaluation_result
            )

            print_test_result(
                index,
                test_case,
                evaluation_result,
            )

        except Exception as exc:

            print()

            print(
                f"TEST CASE {index} FAILED "
                f"WITH EXCEPTION"
            )

            print(
                f"Error: {exc}"
            )

            results.append(
                {
                    "tool": False,
                    "arguments": False,
                    "tool_data": False,
                    "final_answer": False,
                    "overall": False,
                    "actual_tool": None,
                    "actual_arguments": {},
                }
            )


    # ========================================================
    # EVALUATION SUMMARY
    # ========================================================

    print()

    print("=" * 75)

    print(
        "EVALUATION SUMMARY"
    )

    print("=" * 75)

    total_tests = len(results)

    tool_passed = sum(
        result["tool"]
        for result in results
    )

    argument_passed = sum(
        result["arguments"]
        for result in results
    )

    data_passed = sum(
        result["tool_data"]
        for result in results
    )

    answer_passed = sum(
        result["final_answer"]
        for result in results
    )

    overall_passed = sum(
        result["overall"]
        for result in results
    )

    print()

    print(
        f"Tool Selection : "
        f"{tool_passed}/{total_tests}"
    )

    print(
        f"Arguments      : "
        f"{argument_passed}/{total_tests}"
    )

    print(
        f"Tool Data      : "
        f"{data_passed}/{total_tests}"
    )

    print(
        f"Final Answer   : "
        f"{answer_passed}/{total_tests}"
    )

    print(
        f"Overall        : "
        f"{overall_passed}/{total_tests}"
    )

    print()

    # --------------------------------------------------------
    # Overall percentage
    # --------------------------------------------------------

    evaluation_percentage = (
        overall_passed
        / total_tests
        * 100
    )

    print(
        f"Evaluation Score: "
        f"{evaluation_percentage:.2f}%"
    )

    print()

    if overall_passed == total_tests:

        print(
            "STATUS: ALL TESTS PASSED"
        )

    else:

        print(
            "STATUS: SOME TESTS FAILED"
        )

    print()

    print("=" * 75)


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        run_evaluation()
    )