import os
import sqlite3

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from tools import get_tools

# Load environment variables from .env file - Open API key
load_dotenv()



# Set Up ReAct Agent with Document Store Retriever
# Load the ReAct Docstore Prompt
# https://smith.langchain.com/hub/hwchase17/react
react_docstore_prompt = hub.pull("hwchase17/react")


# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=get_tools(),
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=get_tools(), handle_parsing_errors=True, verbose=True,
)

chat_history = []
while True:

    query = input("\nYou: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke(
        {"input": query, "chat_history": chat_history})
    print(f"Agent: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))
