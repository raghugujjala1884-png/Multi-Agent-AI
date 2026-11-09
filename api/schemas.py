from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question submitted by the user"
    )


class ChatResponse(BaseModel):
    question: str
    answer: str
    status: str