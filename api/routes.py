from fastapi import APIRouter
from langchain_core.messages import HumanMessage

from api.schemas import ChatRequest, ChatResponse
from multi_agent.multi_agent import app as graph_app


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    question = request.question.strip()

    result = await graph_app.ainvoke({
        "messages": [HumanMessage(content=question)],
        "user_question": question,
        "completed_agents": []
    })

    agent_messages = result.get("messages", [])[1:]

    answer = "\n".join(
        str(message.content)
        for message in agent_messages
        if message.content
    )

    return ChatResponse(
        question=question,
        answer=answer,
        status="success"
    )