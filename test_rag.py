import asyncio
from multi_agent.rag_agent import rag_agent as run 


async def main():
    question = "Summarize the climate information in the retrieved documents."

    answer = await run(question)

    print("\n========== RAG TEST RESULT ==========")
    print(answer)
    print("=====================================")


if __name__ == "__main__":
    asyncio.run(main())