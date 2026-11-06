from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.tools import StructuredTool


class CalculatorInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    # operation: str = Field(
    #     description="Operation to perform: add, subtract, multiply, or divide"
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
    description="Operation to perform"
    )
    


def calculate(a: float, b: float, operation: str):
    """Perform addition, subtraction, multiplication, or division on two numbers."""

    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            return "Cannot divide by zero."

    if operation == "divide":
        return a / b

    return "Unsupported operation."


calculator = StructuredTool.from_function(
    func=calculate,
    name="calculator",
    description="Perform basic arithmetic calculations.",
    args_schema=CalculatorInput
)