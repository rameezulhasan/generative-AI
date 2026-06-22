import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

## Prompt Template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user","Question: {question}")
    ])

## Streamlit framework
st.title("This is my langchain Demo with Qwen 2.5")
input_text = st.text_input("What question you have in mind?")

## Ollama Qwen2.5
llm= Ollama(model="qwen2.5:1.5b")

output_Parser = StrOutputParser()
chain = prompt|llm|output_Parser
if input_text:
    st.write(chain.invoke({"question",input_text}))
