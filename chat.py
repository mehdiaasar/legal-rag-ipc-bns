import os
# Set OPENAI_API_KEY in your environment or a .env file (see .env.example)
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
import streamlit as st


persist_directory='ip+bns_pdf'

embedding = OpenAIEmbeddings()

vectordb=Chroma(persist_directory=persist_directory,
                embedding_function=embedding)
from langchain_community.llms import Ollama
## Load Ollama LAMA2 LLM model
llm = OpenAI(temperature=0.3)
# llm =Ollama(model="llama3")

## Design ChatPrompt Template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
<context>
{context}
</context>
Question: {input}
"""
)
from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain=create_stuff_documents_chain(llm,prompt)

retriever=vectordb.as_retriever()

from langchain.chains import create_retrieval_chain
retrieval_chain=create_retrieval_chain(retriever,document_chain)
# chian =retrieval_chain|StrOutputParser()
st.title("lanchain Demo with OPENAI API")
input_text=st.text_input('Search the topic u want')
if input_text:
    response=retrieval_chain.invoke({"input":input_text})
    st.write(response["answer"])
