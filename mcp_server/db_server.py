import sqlite3

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Company Database Server")


@mcp.tool()
def get_employee(employee_id: str) -> dict:
    """
    Get employee details using employee ID.
    """

    connection = sqlite3.connect("database/company.db")
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT employee_id, name, department, designation, salary
        FROM employees
        WHERE employee_id = ?
        """,
        (employee_id,)
    )

    employee = cursor.fetchone()

    connection.close()

    if employee is None:
        return {
            "error": f"Employee {employee_id} was not found."
        }

    return {
        "employee_id": employee[0],
        "name": employee[1],
        "department": employee[2],
        "designation": employee[3],
        "salary": employee[4]
    }


@mcp.tool()
def get_employees_by_department(department: str) -> list[dict]:
    """
    Get all employees belonging to a department.
    """

    connection = sqlite3.connect("database/company.db")
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT employee_id, name, department, designation, salary
        FROM employees
        WHERE department = ?
        """,
        (department,)
    )

    employees = cursor.fetchall()

    connection.close()

    result = []

    for employee in employees:
        result.append({
            "employee_id": employee[0],
            "name": employee[1],
            "department": employee[2],
            "designation": employee[3],
            "salary": employee[4]
        })

    return result


@mcp.tool()
def search_employee_by_name(name: str) -> list[dict]:
    """
    Search employees using their name.
    """

    connection = sqlite3.connect("database/company.db")
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT employee_id, name, department, designation, salary
        FROM employees
        WHERE name LIKE ?
        """,
        (f"%{name}%",)
    )

    employees = cursor.fetchall()

    connection.close()

    result = []

    for employee in employees:
        result.append({
            "employee_id": employee[0],
            "name": employee[1],
            "department": employee[2],
            "designation": employee[3],
            "salary": employee[4]
        })

    return result


if __name__ == "__main__":
    mcp.run()