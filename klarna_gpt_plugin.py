from dotenv import load_dotenv as load_env
from langchain.chat_models import ChatOpenAI as chat
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType as at
from langchain.tools import AIPluginTool as tool

load_env()

llm = chat(temperature=0)
tools = load_tools(["requests_all"])
# tool = tool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")
tools += [tool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")]

agent_chain = initialize_agent(
    tools, llm, agent=at.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent_chain.run("what black color t-sharts are available in klrna?")


# msg = "Roll a dice"
# print(msg)

# Retrying langchain.chat_models.openai.ChatOpenAI.completion_with_retry.<locals>._completion_with_retry in 4.0 seconds as it raised RateLimitError: You exceeded your current quota, please check your plan and billing details..
