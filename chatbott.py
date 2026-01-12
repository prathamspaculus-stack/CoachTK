from dotenv import load_dotenv
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated 
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import mysql.connector


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

all_data = vectorstore.get()

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

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

mysql_conn = mysql.connector.connect(
    host="localhost",
    user="coach_user",
    password="coach123",
    database="coachtk"
)

mysql_cursor = mysql_conn.cursor()


class CoachAnswer(TypedDict):
    messages : Annotated[List[BaseMessage], add_messages]
    retrieved_docs: List[Document]
    context: str
    answer: str

def retrieve_docs(state: CoachAnswer ):

    last_question = state["messages"][-1].content
    docs = retriever.invoke(last_question)

    return {"retrieved_docs": docs}

def context(state: CoachAnswer):

    chat_history = "\n".join(
        f"{m.type.upper()}: {m.content}" for m in state["messages"]
    )

    blocks = []

    
    for i, doc in enumerate(state["retrieved_docs"], 1):
        text = doc.page_content
        link = doc.metadata.get("reference_link")

        block = f"""
CONTENT {i}:
{text}
"""
        if link:
            block += f"\nSOURCE LINK: {link}"

        blocks.append(block.strip())

    context_text = f"""
CHAT HISTORY:
{chat_history}

CONTENT:
{"\n\n---\n\n".join(blocks)}
"""

    return {"context": context_text}


def save_to_mysql(thread_id, question, answer, chunk):
    sql = """
    INSERT INTO coach_chat_logs
    (thread_id, user_question, ai_answer, chunk)
    VALUES (%s, %s, %s, %s)
    """
    mysql_cursor.execute(sql, (thread_id, question, answer, chunk))
    mysql_conn.commit()


def answer(state: CoachAnswer):
    question = state["messages"][-1].content

    prompt = p1.format(
        content=state["context"],
        question=question
    )

    response = model.invoke(prompt)
    final_answer = parser.invoke(response)

    # 🔹 Combine chunks (retrieved docs)
    chunks = "\n\n---\n\n".join(
        doc.page_content for doc in state["retrieved_docs"]
    )

    # 🔹 thread_id (same as workflow config)
    thread_id = "1"  # later you can make dynamic

    # 🔹 SAVE EVERYTHING
    save_to_mysql(
        thread_id=thread_id,
        question=question,
        answer=final_answer,
        chunk=chunks
    )

    return {"answer": final_answer}


graph = StateGraph(CoachAnswer)

graph.add_node('retrieve_docs', retrieve_docs)
graph.add_node('context', context)
graph.add_node('answer', answer)

graph.add_edge(START, 'retrieve_docs')
graph.add_edge('retrieve_docs', 'context')
graph.add_edge('context', 'answer')
graph.add_edge('answer', END)

conn = sqlite3.connect(database='chat.db', check_same_thread=False)
cursor = conn.cursor()
checkpointer = SqliteSaver(conn=conn)

workflow = graph.compile(checkpointer=checkpointer)


while True:
    user_message = input("you: ")

    if user_message.lower() in ['exit', 'quite', 'bye']:
        break

    config1 = {"configurable": {"thread_id": "1"}}

    response = workflow.invoke({'messages': [HumanMessage(content=user_message)]}, config=config1)

    print('TK:', response['answer'])

# print(workflow.get_state(config1))

# initial_state = { 
#     "messages" : [HumanMessage(content="tell me about the topics like entrepreneurship and innovation")
#     ]}

# config1 = {"configurable": {"thread_id": "1"}}

# result = workflow.invoke(initial_state, config=config1)
# print(result['answer'])

# initial_state1 = {
#     "messages": [HumanMessage(content="tell me my name")]
# }

# # config2 = {"configurable": {"thread_id": "2"}}

# result1 = workflow.invoke(initial_state1, config=config1)
# print(result1['answer'])