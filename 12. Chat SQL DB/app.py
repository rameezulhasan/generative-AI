import streamlit as st
from pathlib import Path
import sqlite3

from sqlalchemy import create_engine

# Latest LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import (
    SQLDatabaseToolkit,
    create_sql_agent,
)
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


# ----------------------------------------------------
# Streamlit Page
# ----------------------------------------------------
st.set_page_config(
    page_title="LangChain Chat with SQL",
    page_icon="🦜",
    layout="wide"
)

st.title("🦜 Chat with SQL Database")


# ----------------------------------------------------
# Constants
# ----------------------------------------------------
LOCAL_DB = "USE_LOCAL_DB"
MYSQL_DB = "USE_MYSQL"


# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
radio_options = [
    "Use SQLite Database (student.db)",
    "Connect MySQL Database"
]

selected_option = st.sidebar.radio(
    "Choose Database",
    radio_options
)


if selected_option == radio_options[0]:
    db_type = LOCAL_DB
else:
    db_type = MYSQL_DB

# ---------------- MySQL Credentials ----------------

mysql_host = None
mysql_user = None
mysql_password = None
mysql_database = None

if db_type == MYSQL_DB:

    mysql_host = st.sidebar.text_input("Host")
    mysql_user = st.sidebar.text_input("Username")
    mysql_password = st.sidebar.text_input(
        "Password",
        type="password"
    )
    mysql_database = st.sidebar.text_input("Database Name")


# ---------------- Groq API ----------------

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)


if not groq_api_key:
    st.info("Please enter your Groq API Key.")
    st.stop()


# ----------------------------------------------------
# LLM
# ----------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    streaming=True,
    temperature=0
)


# ----------------------------------------------------
# Configure Database
# ----------------------------------------------------
@st.cache_resource(ttl="2h")
def configure_database(
    db_type,
    host=None,
    user=None,
    password=None,
    database=None
):

    if db_type == LOCAL_DB:

        db_path = (
            Path(__file__).parent /
            "student.db"
        ).absolute()

        creator = lambda: sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True
        )

        engine = create_engine(
            "sqlite:///",
            creator=creator
        )

        return SQLDatabase(engine)


    elif db_type == MYSQL_DB:

        if not all([host, user, password, database]):
            st.error("Please provide all MySQL credentials.")
            st.stop()

        connection_string = (
            f"mysql+mysqlconnector://"
            f"{user}:{password}@{host}/{database}"
        )

        engine = create_engine(connection_string)

        return SQLDatabase(engine)


# ----------------------------------------------------
# Create Database Object
# ----------------------------------------------------
if db_type == LOCAL_DB:

    db = configure_database(db_type)

else:

    db = configure_database(
        db_type,
        mysql_host,
        mysql_user,
        mysql_password,
        mysql_database
    )


# ----------------------------------------------------
# SQL Toolkit
# ----------------------------------------------------
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

# ----------------------------------------------------
# Create SQL Agent
# ----------------------------------------------------
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)


# ----------------------------------------------------
# Chat History
# ----------------------------------------------------
if (
    "messages" not in st.session_state
    or st.sidebar.button("Clear Chat History")
):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me anything about your SQL database."
        }
    ]


# ----------------------------------------------------
# Display Previous Messages
# ----------------------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ----------------------------------------------------
# User Input
# ----------------------------------------------------
user_query = st.chat_input(
    "Ask anything from your SQL database..."
)


# ----------------------------------------------------
# Run Agent
# ----------------------------------------------------
if user_query:

    # Show User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant Response
    with st.chat_message("assistant"):

        st_callback = StreamlitCallbackHandler(
            st.container()
        )

        try:

            response = agent.invoke(
                {"input": user_query},
                {
                    "callbacks": [st_callback]
                }
            )

            answer = response["output"]

        except Exception as e:

            answer = f"❌ Error: {str(e)}"

        st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
        