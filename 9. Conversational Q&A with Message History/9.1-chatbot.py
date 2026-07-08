import os
import uuid
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ------------------------
# Load Environment
# ------------------------
load_dotenv()

st.set_page_config(
    page_title="AI Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Chat")

# ------------------------
# Session State
# ------------------------
if "chats" not in st.session_state:
    first_chat = str(uuid.uuid4())

    st.session_state.chats = {
        first_chat: {
            "title": "Chat 1",
            "messages": [],
            "retriever": None,
            "pdf_name": None
        }
    }

    st.session_state.current_chat = first_chat

# ------------------------
# Embeddings (Ollama - loaded once)
# ------------------------
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )

embeddings = get_embeddings()

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:

    st.header("Chats")

    if st.button("➕ New Chat"):

        chat_id = str(uuid.uuid4())

        chat_number = len(st.session_state.chats) + 1

        st.session_state.chats[chat_id] = {
            "title": f"Chat {chat_number}",
            "messages": [],
            "retriever": None,
            "pdf_name": None
        }

        st.session_state.current_chat = chat_id

        st.rerun()

    st.divider()

    # ------------------------
    # Chat List with Delete
    # ------------------------
    for chat_id, chat in list(st.session_state.chats.items()):

        col1, col2 = st.columns([4, 1])

        with col1:
            if st.button(chat["title"], key=chat_id):

                st.session_state.current_chat = chat_id

                st.rerun()

        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}"):

                del st.session_state.chats[chat_id]

                if len(st.session_state.chats) == 0:

                    new_chat_id = str(uuid.uuid4())

                    st.session_state.chats[new_chat_id] = {
                        "title": "Chat 1",
                        "messages": [],
                        "retriever": None,
                        "pdf_name": None
                    }

                    st.session_state.current_chat = new_chat_id

                elif st.session_state.current_chat == chat_id:

                    st.session_state.current_chat = list(
                        st.session_state.chats.keys()
                    )[0]

                st.rerun()

    st.divider()

    # ------------------------
    # PDF Upload (per chat)
    # ------------------------
    st.subheader("📄 PDF Knowledge Base")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF for this chat",
        type=["pdf"],
        key=f"uploader_{st.session_state.current_chat}"
    )

    current_chat_ref = st.session_state.chats[st.session_state.current_chat]

    if uploaded_pdf is not None and current_chat_ref["pdf_name"] != uploaded_pdf.name:

        with st.spinner("Reading and indexing PDF..."):

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.getvalue())
                tmp_path = tmp_file.name

            # Load PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Chunking
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

            chunks = splitter.split_documents(pages)

            # Embeddings + Chroma Vector Store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=st.session_state.current_chat
            )

            # Retriever
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            current_chat_ref["retriever"] = retriever
            current_chat_ref["pdf_name"] = uploaded_pdf.name

            os.remove(tmp_path)

        st.success(f"✅ {uploaded_pdf.name} indexed!")

    if current_chat_ref["pdf_name"]:
        st.info(f"📎 Active PDF: {current_chat_ref['pdf_name']}")

# ------------------------
# Current Chat
# ------------------------
current_chat = st.session_state.chats[
    st.session_state.current_chat
]

messages = current_chat["messages"]

# ------------------------
# Show History
# ------------------------
for message in messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.write(message.content)

    else:

        with st.chat_message("assistant"):
            st.write(message.content)

# ------------------------
# LLM
# ------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5
)

# Normal chat prompt (no PDF)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant."
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{question}"
        )
    ]
)

# RAG prompt (with PDF context)
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant. Use the context below to answer the question.
If the answer is not found in the context, say you don't know based on the document.

Context:
{context}
"""
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{question}"
        )
    ]
)

parser = StrOutputParser()

chain = prompt | llm | parser
rag_chain = rag_prompt | llm | parser

def generate_chat_title(llm, first_message):

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Generate a chat title.

Rules:

- Maximum 4 words.
- No quotation marks.
- No punctuation.
- Return only the title.
"""
            ),

            (
                "human",
                "{message}"
            )
        ]
    )

    parser = StrOutputParser()

    title_chain = title_prompt | llm | parser

    title = title_chain.invoke(
        {
            "message": first_message
        }
    )

    return title.strip()

# ------------------------
# User Input
# ------------------------
user_input = st.chat_input("Ask anything...")


if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    # Previous history only
    history = messages.copy()

    retriever = current_chat.get("retriever")

    if retriever:

        # ------------------------
        # RAG Flow
        # ------------------------
        retrieved_docs = retriever.invoke(user_input)

        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

        response = rag_chain.invoke(
            {
                "context": context,
                "chat_history": history,
                "question": user_input
            }
        )

    else:

        # ------------------------
        # Normal Flow
        # ------------------------
        response = chain.invoke(
            {
                "chat_history": history,
                "question": user_input
            }
        )

    if len(messages) == 0:
        current_chat["title"] = generate_chat_title(
        llm,
        user_input
    )

    with st.chat_message("assistant"):
        st.write(response)


    messages.append(HumanMessage(content=user_input))
    messages.append(AIMessage(content=response))