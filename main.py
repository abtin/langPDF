import base64

from ui_templates import css_style, ai_template, user_template

import os

import streamlit as st
from loguru import logger as log
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def init():
    log.info("Initializing the app")
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        log.error("OPENAI_API_KEY is not set")
        exit(1)
    st.set_page_config("Market Whisperer", page_icon="💬", layout="wide")
    st.write(css_style, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


def load_images(user_image_file, robot_image_file):
    user_image = open(user_image_file, "rb")
    robot_image = open(robot_image_file, "rb")
    user_url = base64.b64encode(user_image.read()).decode()
    robot_url = base64.b64encode(robot_image.read()).decode()
    user_image.close()
    robot_image.close()
    return user_url, robot_url


def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    for pdf in pdf_file:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def create_chunks(text: str) -> list[str]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_db(chunks: list[str]) -> Chroma:
    return Chroma.from_texts(texts=chunks, embedding=OpenAIEmbeddings(), persist_directory="./embeddings")


def create_conversation_chain(vectordb: Chroma):
    buffered_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chatllm = ChatOpenAI(temperature=0)
    first_message = [
        SystemMessage(content="Only answer questions about the uploaded PDFs. "+
                              "Refuse to answer any questions out of the context. "+
                              "For every answer, provide the page on the PDF for reference. "+
                              "If you don't know the answer, say I don't know.")
    ]
    chatllm(first_message)
    chain = ConversationalRetrievalChain.from_llm(
        llm=chatllm,
        retriever=vectordb.as_retriever(),
        memory=buffered_memory
    )
    return chain


def generate_answer(query: str):
    response = st.session_state.conversation({"question": query})
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template
                     .replace("{USER_MSG}", message.content)
                     .replace("{USER_IMAGE}", st.session_state.user_image),
                     unsafe_allow_html=True)
        else:
            st.write(ai_template
                     .replace("{AI_MSG}", message.content)
                     .replace("{ROBOT_IMAGE}", st.session_state.robot_image),
                     unsafe_allow_html=True)


def main():
    init()
    st.session_state.user_image, st.session_state.robot_image = load_images("static/person.png", "static/robot.png")
    st.title("Market Whisperer 💬")
    query = st.text_input("Enter your question here:")
    if query != "":
        generate_answer(query)

    with st.sidebar:
        st.subheader("Annual Reports")
        pdf_files = st.file_uploader("Upload annual reports:", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit"):
            with st.spinner("Processing..."):
                text = extract_text_from_pdf(pdf_files)
                chunks = create_chunks(text)
                vector_db = create_vector_db(chunks)
                st.session_state.conversation = create_conversation_chain(vector_db)


if __name__ == '__main__':
    main()
