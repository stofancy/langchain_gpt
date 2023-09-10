"""LLM：从语言模型中输出预测结果，和直接使用OpenAI的接口一样，输入什么就返回什么。 """
from langchain import OpenAI
llm = OpenAI(model_name='text-davinci-003', temperature=0.9)
text = 'What whould be a good company name for a company that provides software development service?'
# use OpenAI directly
print(llm(text))
