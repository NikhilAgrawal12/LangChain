from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a paragraph about {topic}",
    input_variables= ["topic"]
)

prompt2 = PromptTemplate(
    template="Write 3 question quiz about {text}",
    input_variables=['text']
)

model = ChatOpenAI(model = "gpt-5-nano")

parser = StrOutputParser()

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(chain.invoke({"topic": "AI"}))  

