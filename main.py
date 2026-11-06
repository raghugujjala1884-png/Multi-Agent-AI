from chatbot.chatbot import Chatbot


print("=" * 50)
print("LangGraph Tool-Calling Chatbot")
print("Type 'exit' to quit")
print("=" * 50)

chatbot = Chatbot()

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("AI: Goodbye!")
        break

    print("AI: ", end="", flush=True)

    for chunk in chatbot.chat(user_input):
        print(chunk, end="", flush=True)

    print()