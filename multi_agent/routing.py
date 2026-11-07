from typing import Literal, TypedDict


class RoutingState(TypedDict):
    next_agent: Literal[
        "employee_agent",
        "weather_agent",
        "rag_agent",
        "FINISH"
    ]


def route_agent(
    state: RoutingState
) -> Literal[
    "employee_agent",
    "weather_agent",
    "rag_agent",
    "FINISH"
]:
    return state["next_agent"]