from dotenv import load_dotenv
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
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from mysql_memory import save_chat, load_chat
import uuid

load_dotenv()

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
- You may also use information shared earlier by the user in this conversation
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
    history = "\n".join(f"{m.type.upper()}: {m.content}" for m in state["messages"])

    blocks = []
    for i, doc in enumerate(state["retrieved_docs"], 1):
        block = f"CONTENT {i}:\n{doc.page_content}"
        blocks.append(block)

    return {
        "context": f"CHAT HISTORY:\n{history}\n\nCONTENT:\n" + "\n\n---\n\n".join(blocks)
    }

def answer(state: CoachAnswer):
    prompt = p1.format(
        content=state["context"],
        question=state["messages"][-1].content
    )
    response = model.invoke(prompt)
    return {"answer": parser.invoke(response)}

graph = StateGraph(CoachAnswer)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("context", context)
graph.add_node("answer", answer)

graph.add_edge(START, "retrieve_docs")
graph.add_edge("retrieve_docs", "context")
graph.add_edge("context", "answer")
graph.add_edge("answer", END)

conn = sqlite3.connect("chat.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
workflow = graph.compile(checkpointer=checkpointer)

THREAD_ID = str(uuid.uuid4())
config = {"configurable": {"thread_id": THREAD_ID}}

while True:
    user_message = input("you: ")

    if user_message.lower() in ["exit", "quit", "bye"]:
        break

    history = load_chat(THREAD_ID)
    messages = []

    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    save_chat(THREAD_ID, "user", user_message)

    response = workflow.invoke(
        {"messages": messages + [HumanMessage(content=user_message)]},
        config=config
    )

    ai_answer = response["answer"]

    save_chat(THREAD_ID, "ai", ai_answer)

    print("TK:", ai_answer)




