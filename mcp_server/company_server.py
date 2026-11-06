from mcp.server.mcpserver import MCPServer


mcp = MCPServer("Company Assistant")


# ============================================================
# SAMPLE COMPANY DATA
# ============================================================

employees = {
    "EMP001": {
        "name": "Ravi",
        "department": "Data Science",
        "role": "Data Scientist"
    },
    "EMP002": {
        "name": "Priya",
        "department": "Engineering",
        "role": "Software Engineer"
    },
    "EMP003": {
        "name": "Arun",
        "department": "HR",
        "role": "HR Manager"
    }
}


# ============================================================
# MCP TOOL
# ============================================================

@mcp.tool()
def get_employee(employee_id: str) -> dict:
    """Get employee information using an employee ID."""

    if employee_id not in employees:
        raise ValueError(
            f"Employee {employee_id} was not found."
        )

    return employees[employee_id]


# ============================================================
# MCP TOOL
# ============================================================

@mcp.tool()
def get_department_employees(department: str) -> list:
    """Get employees belonging to a department."""

    result = []

    for employee_id, employee in employees.items():

        if employee["department"].lower() == department.lower():

            result.append({
                "employee_id": employee_id,
                **employee
            })

    return result


# ============================================================
# MCP RESOURCE
# ============================================================

@mcp.resource("company://leave-policy")
def leave_policy() -> str:
    """Company leave policy."""

    return """
Company Leave Policy

1. Employees receive 24 annual leave days.
2. Employees can request leave through the HR system.
3. Leave requests should normally be submitted in advance.
4. Emergency leave may be requested when necessary.
5. Manager approval is required for planned leave.
6. Unused leave is subject to company carry-forward rules.
"""


# ============================================================
# MCP RESOURCE
# ============================================================

@mcp.resource("company://working-hours")
def working_hours() -> str:
    """Company working hours."""

    return """
Company Working Hours

Monday to Friday
9:00 AM to 6:00 PM

Lunch:
1:00 PM to 2:00 PM

Employees should follow their team-specific working
arrangements where applicable.
"""


# ============================================================
# MCP PROMPT
# ============================================================

@mcp.prompt()
def explain_leave_policy(audience: str = "employee") -> str:
    """Create a prompt for explaining the company leave policy."""

    return f"""
Explain the company leave policy to a {audience}.

Use simple language.

Cover:
1. Annual leave entitlement
2. Leave request process
3. Emergency leave
4. Manager approval
5. Carry-forward rules

Do not invent policies that are not provided.
"""