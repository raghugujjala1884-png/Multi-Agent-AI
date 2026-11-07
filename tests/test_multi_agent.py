from multi_agent.multi_agent import route_agent


def test_route_agent_employee():
    state = {
        "next_agent": "employee_agent"
    }

    result = route_agent(state)

    assert result == "employee_agent"


def test_route_agent_weather():
    state = {
        "next_agent": "weather_agent"
    }

    result = route_agent(state)

    assert result == "weather_agent"


def test_route_agent_rag():
    state = {
        "next_agent": "rag_agent"
    }

    result = route_agent(state)

    assert result == "rag_agent"