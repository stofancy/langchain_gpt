"""Agents：基于用户输入动态地调用chains，LangChani可以将问题拆分为几个步骤，然后每个步骤可以根据提供个Agents做相关的事情。 """
from langchain import OpenAI
from langchain.agents import initialize_agent, load_tools, AgentType

llm = OpenAI(model_name='text-davinci-003', temperature=0.9)
#  Agents：基于用户输入动态地调用chains，LangChain可以将问题拆分为几个步骤，然后每个步骤可以根据提供个Agents做相关的事情。
tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# text = "The principal is 2,490,000, and calculated at a simple interest rate of 5.55% for thirty years, what is the interest?"
text = "12 raised to the 3 power and result raised to 2 power?"
print("input text:", text)
agent.run(text)
