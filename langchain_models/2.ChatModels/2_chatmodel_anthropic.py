from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-v1')

result = model.invoke("What is capital of India")

print(result.content)
