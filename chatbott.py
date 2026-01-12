from dotenv import load_dotenv
import uuid
import mysql.connector

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

load_dotenv()

THREAD_ID = str(uuid.uuid4())
print("New thread started:", THREAD_ID)

parser = StrOutputParser()

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

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

p1 = PromptTemplate(
    input_variables=["question", "content"],
    template="""
You are a practical coach Terry kim, not a teacher.

RULES:
- Use the provided content as PRIMARY source
- Do NOT add outside knowledge
- Keep language simple and practical

CONTEXT:
{content}

USER QUESTION:
{question}

ANSWER:
"""
)

mysql_conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="newpassword",
    database="coachtk"
)
mysql_cursor = mysql_conn.cursor()

def save_qa(thread_id, question, answer, chunk):
    sql = """
    INSERT INTO coach_chat_logs
    (thread_id, user_question, ai_answer, chunk)
    VALUES (%s, %s, %s, %s)
    """
    mysql_cursor.execute(sql, (thread_id, question, answer, chunk))
    mysql_conn.commit()

def load_history(thread_id, limit=5):
    sql = """
    SELECT user_question, ai_answer
    FROM coach_chat_logs
    WHERE thread_id = %s
    ORDER BY created_at
    LIMIT %s
    """
    mysql_cursor.execute(sql, (thread_id, limit))
    rows = mysql_cursor.fetchall()

    messages = []
    for q, a in rows:
        messages.append(HumanMessage(content=q))
        messages.append(AIMessage(content=a))

    return messages

class CoachAnswer(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retrieved_docs: List[Document]
    context: str
    answer: str

def retrieve_docs(state: CoachAnswer):
    question = state["messages"][-1].content
    docs = retriever.invoke(question)
    return {"retrieved_docs": docs}

def context(state: CoachAnswer):
    chat_history = "\n".join(
        f"{m.type.upper()}: {m.content}" for m in state["messages"]
    )

    chunks = []
    for i, doc in enumerate(state["retrieved_docs"], 1):
        chunks.append(f"CONTENT {i}:\n{doc.page_content}")

    return {
        "context": f"""
CHAT HISTORY:
{chat_history}

CONTENT:
{"\n\n---\n\n".join(chunks)}
"""
    }

def answer(state: CoachAnswer):
    question = state["messages"][-1].content

    prompt = p1.format(
        content=state["context"],
        question=question
    )

    response = model.invoke(prompt)
    final_answer = parser.invoke(response)

    chunk_text = "\n\n---\n\n".join(
        doc.page_content for doc in state["retrieved_docs"]
    )

    save_qa(THREAD_ID, question, final_answer, chunk_text)

    return {"answer": final_answer}

graph = StateGraph(CoachAnswer)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("context", context)
graph.add_node("answer", answer)

graph.add_edge(START, "retrieve_docs")
graph.add_edge("retrieve_docs", "context")
graph.add_edge("context", "answer")
graph.add_edge("answer", END)

workflow = graph.compile()

while True:
    user_message = input("You: ")

    if user_message.lower() in ["exit", "quit", "bye"]:
        print("Session ended.")
        break

    past_messages = load_history(THREAD_ID)

    state = {
        "messages": past_messages + [HumanMessage(content=user_message)]
    }

    result = workflow.invoke(state)

    print("TK:", result["answer"])
