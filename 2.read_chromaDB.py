from dotenv import load_dotenv
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="podcast_chunks"
)

vectorstore.get()

collection = vectorstore._collection

data = collection.get(
    include=["documents", "metadatas", "embeddings"]
)


print("\nTEXT:")
print(data["documents"][2209])  

print("\nMETADATA:")
print(data["metadatas"][2209])

print("\nEMBEDDING (first 10 values):")
print(data["embeddings"][2209][:10])

print("\nEMBEDDING LENGTH:")
print(len(data["embeddings"][2209]))