from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("langchain_text_splitters/linear_regression.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0,
)

chunks = splitter.split_text(docs[0].page_content)

print(len(chunks))
print(chunks) 