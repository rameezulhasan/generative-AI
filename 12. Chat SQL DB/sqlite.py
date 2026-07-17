import sqlite3


DATABASE_NAME = "student.db"


def create_database():
    """
    Create SQLite database and STUDENT table.
    Insert sample data only if the table is empty.
    """

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Create STUDENT table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS STUDENT (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            city TEXT,
            course TEXT
        )
    """)

    # Check if table already contains data
    cursor.execute("SELECT COUNT(*) FROM STUDENT")
    count = cursor.fetchone()[0]

    if count == 0:

        students = [
            ("Ali", 21, "Lahore", "AI"),
            ("Ahmed", 22, "Karachi", "Data Science"),
            ("Sara", 20, "Islamabad", "Cyber Security"),
            ("Fatima", 23, "Faisalabad", "Software Engineering"),
            ("Usman", 24, "Multan", "AI"),
            ("Ayesha", 22, "Lahore", "Machine Learning"),
            ("Bilal", 25, "Peshawar", "Computer Science"),
            ("Hina", 21, "Rawalpindi", "AI"),
            ("Hamza", 23, "Sialkot", "Information Technology"),
            ("Zain", 20, "Faisalabad", "Data Science")
        ]

        cursor.executemany("""
            INSERT INTO STUDENT(name, age, city, course)
            VALUES (?, ?, ?, ?)
        """, students)


    conn.commit()
    conn.close()

import pandas as pd

def get_all_students():
    conn = sqlite3.connect(DATABASE_NAME)
    query = "SELECT * FROM STUDENT"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_total_students():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM STUDENT")
    total = cursor.fetchone()[0]
    conn.close()
    return total

def get_ai_students():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*)
        FROM STUDENT
        WHERE course='AI'
    """)
    total = cursor.fetchone()[0]
    conn.close()
    return total