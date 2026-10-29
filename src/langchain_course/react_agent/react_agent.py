from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_react import ReactAgent

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
import os
from pydantic import SecretStr
from tavily import TavilyClient

from langchain_tavily import TavilySearch
# tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

load_dotenv()

@tool
def search(query: str):
    """
    A search tool that simulates searching for information.
    Args:
        query (str): The search query.
    Returns:
        dict: A search result.
    """
    print(f"Searching for: {query}")
    # return f"Search results for '{query}'"
    return tavily.search(query=query)

def build_react_agent():
    groq_key: str | None = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(model="qwen/qwen3.8-27b", api_key=SecretStr(groq_key) if groq_key else None, temperature=0.5)
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools)
    return agent

def call_react_agent():
    prompt_template = PromptTemplate(
        input_variables=["info"],
        template="""
        You are a React Agent. Your task is to summarize the following information about Krishna in 3 sentences and define him in 1 sentence.
        Information: {info}
        """,
    )

    agent = build_react_agent()
    response = agent.invoke({"messages": [
        HumanMessage(content="Search for 3 job postings for an AI Engineer in India using linkedin and list them.")
    ]})

    print(response)


if __name__ == "__main__":
    call_react_agent()