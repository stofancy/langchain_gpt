from langchain import OpenAI, ConversationChain
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name='text-davinci-003', temperature=0.9)
prompt = PromptTemplate(input_variables=["service"],
                        template="What whould be a good company name for a company that provides {service}?")
# text = 'What whould be a good company name for a company that provides software development service?'

# use OpenAI directly
# print(llm(text))

# boxing a prompt and then use OpenAI
# chain = LLMChain(llm=llm, prompt=prompt)
# response = chain.run("STEAM education")
# print(response)

#  Agents：基于用户输入动态地调用chains，LangChani可以将问题拆分为几个步骤，然后每个步骤可以根据提供个Agents做相关的事情。
# tools = load_tools(["llm-math"], llm=llm)
# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# # text = "The principal is 2,490,000, and calculated at a simple interest rate of 5.55% for thirty years, what is the interest?"
# text = "12 raised to the 3 power and result raised to 2 power?"
# print("input text:", text)
# agent.run(text)

# Memory：就是提供对话的上下文存储，可以使用Langchain的ConversationChain，在LLM交互中记录交互的历史状态，并基于历史状态修正模型预测。
