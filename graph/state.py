# from typing import TypedDict
# class Chatstate(TypedDict):
#     messages: list
#     # response: str

from langgraph.graph import MessagesState

class Chatstate(MessagesState):
    pass