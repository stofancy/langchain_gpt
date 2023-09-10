
"""
Chains：将LLMs和prompts结合起来，前面提到提供了OpenAI的封装和你需要问的字符串模板，就可以执行获得返回了。
Prompt Templates：管理LLMs的Prompts，就像我们需要管理变量或者模板一样。
 """
from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name='text-davinci-003', temperature=0.9)
prompt = PromptTemplate(input_variables=["service"],
                        template="What whould be a good company name for a company that provides {service}?")

# boxing a prompt and then use OpenAI
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run("STEM Education")
print(response)
