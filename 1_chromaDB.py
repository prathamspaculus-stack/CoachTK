from dotenv import load_dotenv
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

load_dotenv()

json_path = r"C:\Users\Administrator\Desktop\pratham\project 3\book4.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

docs = []
for item in data:
    docs.append(
        Document(
            page_content=item["text"],
            metadata=item.get("metadata", {})
        )
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

split_docs = text_splitter.split_documents(docs)
print(f"Total chunks: {len(split_docs)}")


split_docs = filter_complex_metadata(split_docs)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="chroma_db",
    collection_name="podcast_chunks"
)

# vectorstore = Chroma(
#     persist_directory="chroma_db",
#     embedding_function=embeddings,
#     collection_name="podcast_chunks"
# )

# adding data 

# vectorstore.add_documents(split_docs)

vectorstore.persist()

print("Chroma vector store created successfully")



# print(vectorstore.get(include =["metadatas", "documents"]))

# #  Access internal Chroma collection (debug only)
# collection = vectorstore._collection

# data = collection.get(
#     include=["documents", "metadatas", "embeddings"]
# )

# print("\n--- FIRST RECORD ---")

# print("\nTEXT:")
# print(data["documents"][1][:300])  

# print("\nMETADATA:")
# print(data["metadatas"][1])

# print("\nEMBEDDING (first 10 values):")
# print(data["embeddings"][1][:10])

# print("\nEMBEDDING LENGTH:")
# print(len(data["embeddings"][1]))
