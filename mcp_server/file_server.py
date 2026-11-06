import csv
from pathlib import Path

from pypdf import PdfReader
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Document Server")


@mcp.tool()
def get_company_policy() -> dict:
    """
    Read and return the company leave policy document.
    """

    with open(
        "documents/company_policy.txt",
        "r",
        encoding="utf-8"
    ) as file:

        content = file.read()

    return {
        "document": "company_policy.txt",
        "content": content
    }


@mcp.tool()
def get_all_csv_employees() -> list[dict]:
    """
    Get all employee records from the CSV file.
    """

    employees = []

    with open(
        "documents/employees.csv",
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for employee in reader:
            employees.append(employee)

    return employees


@mcp.tool()
def search_csv_employee(name: str) -> list[dict]:
    """
    Search employees by name in the CSV file.
    """

    employees = []

    with open(
        "documents/employees.csv",
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for employee in reader:

            if name.lower() in employee["name"].lower():
                employees.append(employee)

    return employees


@mcp.tool()
def get_csv_employees_by_department(
    department: str
) -> list[dict]:
    """
    Get employees from a specific department.
    """

    employees = []

    with open(
        "documents/employees.csv",
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for employee in reader:

            if employee["department"].lower() == department.lower():
                employees.append(employee)

    return employees


@mcp.tool()
def get_resume_content() -> dict:
    """
    Read the resume PDF and return its text content.
    """

    pdf_path = "documents/resume.pdf"

    reader = PdfReader(pdf_path)

    content = ""

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        content += (
            f"\n\n========== PAGE {page_number} ==========\n\n"
        )

        if text:
            content += text

    return {
        "document": "resume.pdf",
        "pages": len(reader.pages),
        "content": content
    }


@mcp.tool()
def get_pdf_content(filename: str) -> dict:
    """
    Read a PDF file from the documents folder.
    """

    pdf_path = Path("documents") / filename

    if not pdf_path.exists():
        return {
            "error": f"File '{filename}' was not found."
        }

    if pdf_path.suffix.lower() != ".pdf":
        return {
            "error": "The provided file is not a PDF."
        }

    reader = PdfReader(pdf_path)

    content = ""

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        content += (
            f"\n\n========== PAGE {page_number} ==========\n\n"
        )

        if text:
            content += text

    return {
        "document": filename,
        "pages": len(reader.pages),
        "content": content
    }


@mcp.tool()
def list_documents() -> list[str]:
    """
    List all available files in the documents folder.
    """

    documents_path = Path("documents")

    files = []

    for file in documents_path.iterdir():

        if file.is_file():
            files.append(file.name)

    return files


if __name__ == "__main__":
    mcp.run()