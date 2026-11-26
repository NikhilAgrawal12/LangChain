from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

url = "https://www.amazon.com/Madewell-Short-Sleeve-Stitch-Interest/dp/B0GD8WB2DP?s=shopbop&ref_=sb_ts&th=1&psc=1"
loader = WebBaseLoader(url)

prompt = PromptTemplate(
    template="Answer the question {question} with the following content: {content}",
    input_variables=["question", "content"]
)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({
    "question": "What is the product described on this page and what is its price?",
    "content": docs[0].page_content
})

print(result)