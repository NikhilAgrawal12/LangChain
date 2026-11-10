from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = "Write a report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Summarize the following text {text}",
    input_variables=["text"]
)

model = ChatOpenAI(model="gpt-5")

parser = StrOutputParser()


report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2,model,parser)),
    (lambda x: len(x.split())>500, RunnablePassthrough()),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

final_chain.invoke({"topic": "Artificial Intelligence"})