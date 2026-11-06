from langchain_ollama import ChatOllama
from tools.calculator import calculate
from tools.weather import get_weather



llm=ChatOllama(
    model="qwen3:4b", temperature=0
)

llm_with_tools=llm.bind_tools([calculate,get_weather])