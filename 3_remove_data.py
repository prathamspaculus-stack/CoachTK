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

all_data = vectorstore.get()

ids = all_data["ids"]

print("Total records:", len(ids))

ids_to_delete = ids[0] 

vectorstore.delete(ids=ids_to_delete)
vectorstore.persist()

print(f"Deleted records")
