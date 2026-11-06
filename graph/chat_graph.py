# from langgraph.graph import StateGraph, START, END
# from graph.state import Chatstate
# from nodes.router import router
# from nodes.responses import greetings, normal
# from nodes.chatbot import chatbot
# from nodes.calculator_node import calculator_node

# builder=StateGraph(Chatstate)
# # builder.add_node("chatbot",chatbot)
# # builder.add_edge(START, "chatbot")
# # builder.add_edge("chatbot", END)

# builder.add_node("calculator_node", calculator_node)
# builder.add_edge(START, "calculator_node")
# builder.add_edge("calculator_node", END)
# # builder.add_node("router",router)
# # builder.add_node("greetings",greetings)
# # builder.add_node("normal",normal)

# # # builder.add_edge(START,"router")
# # builder.add_conditional_edges(START,router,{"greetings":"greetings","normal":"normal"})

# # builder.add_edge("greetings", END)
# # builder.add_edge("normal", END)

# graph=builder.compile()
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from graph.state import Chatstate
from nodes.chatbot import chatbot
from tools.calculator import calculate
from tools.weather import get_weather


builder = StateGraph(Chatstate)

builder.add_node("chatbot", chatbot)

tool_node = ToolNode([calculate, get_weather])
builder.add_node("tools", tool_node)

builder.add_edge(START, "chatbot")

builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

builder.add_edge("tools", "chatbot")

graph = builder.compile()