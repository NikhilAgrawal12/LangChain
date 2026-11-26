from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOpenAI(model = "gpt-5-nano")

model2 = ChatOpenAI(model = "gpt-5-mini")

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text \n {text}",
    input_variables = ["text"]
)

prompt2 = PromptTemplate(
    template = "Generate 5 short question answers from the following text \n {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables = ["notes","quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes" : prompt1 | model1 | parser,
    "quiz" : prompt2 | model2 | parser
})

merger_chain = prompt3 | model2 | parser

chain = parallel_chain | merger_chain

text = """
Machine learning is a branch of Artificial Intelligence (AI) that enables computers to learn from data, identify patterns, and make predictions or decisions without being explicitly programmed for every task. Instead of following predefined instructions, machine learning algorithms use historical data to identify relationships and improve their performance over time. The more relevant data a model receives, the better it can become at recognizing patterns and making accurate predictions.

Machine learning is generally divided into three major categories: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled datasets to train models for tasks such as predicting house prices, identifying spam emails, and detecting fraudulent transactions. Unsupervised learning works with unlabeled data to discover hidden patterns and relationships, such as grouping customers based on their purchasing behavior using clustering algorithms. Reinforcement learning involves an agent learning through trial and error by interacting with an environment and receiving rewards or penalties based on its actions.

Another important aspect of machine learning is model training and evaluation. During training, algorithms learn patterns from historical data by adjusting their internal parameters to minimize prediction errors. Once trained, models are evaluated using separate test data to measure their ability to make predictions on previously unseen information. Common machine learning algorithms include Linear Regression, Logistic Regression, Decision Trees, Random Forest, Support Vector Machines, K-Means Clustering, and Neural Networks.

Machine learning has become an essential technology across various industries. It is used in recommendation systems to suggest products and movies, in banking to detect fraudulent transactions, in healthcare to assist with disease prediction, in autonomous vehicles to recognize objects, and in natural language processing to understand and generate human language. As data availability and computational capabilities continue to grow, machine learning plays an increasingly important role in building intelligent applications and automating complex tasks."""

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()



