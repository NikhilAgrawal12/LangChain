from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-5', temperature=0, max_completion_tokens=30)

result = model.invoke("Write 5 lines poem on cricket")

print(type(result))

print(result.content)


