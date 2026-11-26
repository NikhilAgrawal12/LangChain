from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

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

paragraph_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    "paragraph" : RunnablePassthrough(),
    "quiz" : RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(paragraph_chain, parallel_chain)
print(final_chain.invoke({"topic": "AI advancements"}))