import mysql.connector

mysql_conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="newpassword",
    database="chat_memory_db"
)

mysql_cur = mysql_conn.cursor()

def save_chat(thread_id, role, content):
    mysql_cur.execute(
        "INSERT INTO chat_history (thread_id, role, content) VALUES (%s, %s, %s)",
        (thread_id, role, content)
    )
    mysql_conn.commit()

def load_chat(thread_id):
    mysql_cur.execute(
        "SELECT role, content FROM chat_history WHERE thread_id=%s ORDER BY id",
        (thread_id,)
    )
    return mysql_cur.fetchall()
