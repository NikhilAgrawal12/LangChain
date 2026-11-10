from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI(model = "gpt-5-nano")

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive','negative'] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables = ["feedback"],
    partial_variables = {'format_instruction' : parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="""
    Write a short, friendly, and professional response to this positive customer feedback:

    {feedback}

    Return only the final response that will be shown directly to the customer.
    Do not provide multiple options, explanations, headings, or suggestions.
    Keep it within 2-3 sentences.
    """,
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="""
    Write a short, empathetic, and professional response to this negative customer feedback:

    {feedback}

    Acknowledge the customer's concern, apologize for the poor experience,
    and offer assistance.

    Return only the final response that will be shown directly to the customer.
    Do not provide multiple options, explanations, headings, or suggestions.
    Keep it within 2-3 sentences.
    """,
    input_variables=["feedback"]
)


chain = (
    RunnableParallel({
    "feedback": RunnableLambda(lambda x: x["feedback"]),
    "sentiment": classifier_chain
    })
    | RunnableBranch(
        (lambda x: x["sentiment"].sentiment == "positive", prompt2 | model | parser),
        (lambda x: x["sentiment"].sentiment == "negative", prompt3 | model | parser),
        RunnableLambda(lambda x: "could not find sentiment")
    )
)

result = chain.invoke({"feedback": "This is a terrible smart phone"})

print(result)

chain.get_graph().print_ascii()


