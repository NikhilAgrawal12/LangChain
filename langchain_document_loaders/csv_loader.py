from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("langchain_document_loaders/insurance.csv")
docs = loader.load()

print(len(docs))
print(docs[0])
