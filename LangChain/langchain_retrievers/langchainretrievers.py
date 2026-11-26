from dotenv import load_dotenv

load_dotenv()

# Wikipedia Retriever 

# from langchain_community.retrievers import WikipediaRetriever

# retriever = WikipediaRetriever(top_k_results=2, lang="en")

# query = "Advancement in AI"

# docs = retriever.invoke(query)

# for(i, doc) in enumerate(docs):

#     print(f"Document {i+1}:")
#     print(doc.page_content)
#     print()

# Vector Store Retrievers

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# documents = [
#     Document(page_content="LangChain helps developers build LLM applications easily."),
#     Document(page_content="Chroma is a vector database optimized for LLM-based search."),
#     Document(page_content="Embeddings convert text into high-dimensional vectors."),
#     Document(page_content="OpenAI provides powerful embedding models."),
# ]

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="chroma_db",
    collection_name = "my_collection"
)

# vectorstore.add_documents(documents)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

results = retriever.invoke("What is Chrome used for?")

for i, doc in enumerate(results):
    print(f"Document {i+1}:")
    print(doc.page_content)
    print()





