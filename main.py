from loguru import logger as log
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def init():
    log.info("Initializing the app")
    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        log.error("OPENAI_API_KEY is not set")
        exit(1)
    st.set_page_config("PDF Whisperer", page_icon="💬", layout="wide")


def main():
    init()
    st.title("PDF Whisperer 💬")

    pdf = st.file_uploader("Upload your PDF file", type=["pdf"])
    if pdf is not None:
        log.info("PDF file uploaded")
        with st.spinner("Thinking..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len)
            chunks = text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings()
            knowledge_base = Chroma.from_texts(texts=chunks, embedding=embeddings)

        messages = st.session_state.get("messages", [])
        if "messages" not in st.session_state:
            st.session_state.messages = [
                SystemMessage(content="You only answer questions about the pdf here"),
            ]

        user_question = st.text_input("Ask a question about your pdf:", key="user_input")

        if user_question:
            st.session_state.messages.append(HumanMessage(content=user_question))
            with st.spinner("Preparing ..."):
                docs = knowledge_base.similarity_search(user_question, k=3)
                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)
                st.session_state.messages.append(AIMessage(content=response))

            for i, msg in enumerate(messages[1:]):
                if i % 2 == 0:
                    message(msg.content, is_user=True, key=str(i) + "_user", avatar_style="croodles")
                else:
                    message(msg.content, is_user=False, key=str(i) + "_ai", avatar_style="bottts")


if __name__ == '__main__':
    main()
