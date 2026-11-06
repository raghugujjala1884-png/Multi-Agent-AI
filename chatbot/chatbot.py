from graph.chat_graph import graph
from utils.cache import ResponseCache
from langchain_core.messages import HumanMessage, AIMessage


class Chatbot:

    def __init__(self):
        self.messages = []
        self.cache = ResponseCache()

    def get_cache_key(self, user_input):
        return user_input.strip().lower().rstrip("?!.,").strip()

    def chat(self, user_input):

        cache_key = self.get_cache_key(user_input)

        # Cache hit
        if self.cache.exists(cache_key):

            answer = self.cache.get(cache_key)

            self.messages.append(
                HumanMessage(content=user_input)
            )

            self.messages.append(
                AIMessage(content=answer)
            )

            yield answer
            return

        # Add user message
        self.messages.append(
            HumanMessage(content=user_input)
        )

        full_response = ""

        for message, metadata in graph.stream(
            {"messages": self.messages},
            stream_mode="messages"
        ):

            if message.type == "ToolMessage":
                continue

            if message.content:
                full_response += message.content
                yield message.content

        self.messages.append(
            AIMessage(content=full_response)
        )

        self.cache.set(
            cache_key,
            full_response
        )