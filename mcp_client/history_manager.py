from langchain_core.messages import HumanMessage


def get_conversation_turns(messages):
    """
    Split messages into complete conversation turns.
    """

    turns = []
    current_turn = []

    for message in messages:

        if isinstance(message, HumanMessage):

            if current_turn:
                turns.append(current_turn)

            current_turn = [message]

        else:

            current_turn.append(message)

    if current_turn:
        turns.append(current_turn)

    return turns


def get_recent_history(messages, max_turns=2):
    """
    Return the most recent complete conversation turns.
    """

    turns = get_conversation_turns(messages)

    recent_turns = turns[-max_turns:]

    result = []

    for turn in recent_turns:
        result.extend(turn)

    return result