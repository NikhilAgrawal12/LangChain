from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()


prompt = PromptTemplate(
    template="Write a paragraph about {topic}",
    input_variables= ["topic"]
)

model = ChatOpenAI(model = "gpt-5-nano")

parser = StrOutputParser()

paragraph_gen_chain = RunnableSequence(prompt,model,parser)


parallel_chain = RunnableParallel({
    "paragraph": RunnablePassthrough(),
    "word_count": RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(paragraph_gen_chain, parallel_chain)


print(final_chain.invoke({"topic" : "AI news last month"}))