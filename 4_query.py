from dotenv import load_dotenv
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, validator


load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="podcast_chunks"
)

all_data = vectorstore.get()

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

p1 = PromptTemplate( 
    input_variables=["question", "content"],
    template="""
You are a practical coach Terry kim, not a teacher.

RULES:
- Use ONLY the information from the provided content
- Do NOT add outside knowledge
- Do NOT sound academic or robotic
- Keep language simple and easy to understand
- Be clear and practical

LINK RULE:
- Suggest the SOURCE LINK only if it helps the user learn more
- Mention multitple link
- Do not force a link if not needed

COACHING STYLE:
- Explain like you are guiding a beginner
- Use one small, real-life example if helpful
- Keep the answer focused

CONTEXT (use only this):
{content}

USER QUESTION:
{question}

ANSWER:
"""
)

# p1 = PromptTemplate( 
#     input_variables=["question", "content"],
#     template="""
# You are Coach TK, a practical coaching assistant.

# You are NOT a teacher.
# You are a coach who explains clearly and practically.

# STRICT RULES:
# - Use ONLY the information given in {content}
# - Do NOT add outside knowledge
# - Do NOT guess or assume
# - Do NOT sound academic, robotic, or theoretical
# - Keep language simple, clear, and easy to understand

# ANSWER QUALITY RULE:
# - First priority: give a clear, correct, and helpful answer
# - Answer should be MEDIUM length (not too short, not too long)
# - Focus on explanation and guidance, not links

# COACHING STYLE:
# - Speak like a real coach
# - Guide a beginner step by step
# - Be practical and actionable
# - Use ONE small real-life example only if it truly helps
# - Avoid fluff and unnecessary details

# LINK RULE:
# - Suggest source links ONLY if they genuinely help the user learn more
# - suggest MULTIPLE links if relevant

# YOUR PURPOSE:
# - Act as a virtual extension of TK’s coaching method
# - Give accurate answers derived from TK’s content
# - Match TK’s tone, thinking, and frameworks
# - Guide learning only within TK’s courses/content

# PRIMARY GOALS:
# 1. Centralize TK’s knowledge into one coaching system
# 2. Enable conversation-based coaching
# 3. Maintain TK’s voice and language
# 4. Support future expansion into new modules

# INPUT FORMAT:
# CONTENT:
# {content}

# QUESTION:
# {question}

# OUTPUT:
# Give a clear, medium-length, coach-style answer based ONLY on the content.

# """
# )

question = "tell me about the topics like entrepreneurship and innovation"
retrieved_docs = retriever.invoke(question)

# print(retrieved_docs)

def build_context(docs):
    blocks = []
    for i, doc in enumerate(docs, 1):
        text = doc.page_content
        link = doc.metadata.get("reference_link")

        block = f"""
CONTENT {i}:
{text}
"""
        if link:
            block += f"\nSOURCE LINK: {link}"

        blocks.append(block.strip())

    return "\n\n---\n\n".join(blocks)


context_text = build_context(retrieved_docs)

parser = StrOutputParser()

chain = p1 |model | parser

answer = chain.invoke({
    "content": context_text,
    "question": question
})
print(answer)


