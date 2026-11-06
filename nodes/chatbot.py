from graph.state import Chatstate
from models.ollama_models import llm_with_tools
from langchain_core.messages import SystemMessage


def chatbot(state: Chatstate):

    system_message = SystemMessage(
    content=(
        "You are a helpful assistant. "
        "Answer general questions normally. "
        "Use the calculator tool for arithmetic calculations. "
        "Use the weather tool when the user asks about weather. "
        "Do not use tools when they are not necessary."
    )
)

    response = llm_with_tools.invoke(
        [system_message] + state["messages"]
    )

    return {
        "messages": state["messages"] + [response]
    }