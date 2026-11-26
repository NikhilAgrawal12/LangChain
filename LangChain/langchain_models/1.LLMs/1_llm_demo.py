#LLM models are no longer supported so we will use ChatModels Only 

from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-5-nano')

result = llm.invoke("What is the capita of India")

print(result)