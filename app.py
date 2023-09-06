import json
import requests
import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate
from consts import CHROMA_SETTINGS, PERSIST_DIRECTORY

def handle_userinput(user_question):
    response = st.session_state.conversation(user_question)

    st.session_state.chat_history.insert(0, response)

    for i, message in enumerate(st.session_state.chat_history):
        upvoteUrl = "javascript:;"
        downvoteUrl = "javascript:;"
        result = message["result"]
        botTemp = bot_template.replace("{{MSG}}", result)
        botTemp = botTemp.replace("{{DOWNVOTE_URL}}", downvoteUrl)
        botTemp = botTemp.replace("{{UPVOTE_URL}}", upvoteUrl)
        st.write(botTemp, unsafe_allow_html=True)

        userTemp = user_template.replace("{{MSG}}", message["query"])
        st.write(userTemp, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        embeddings = OpenAIEmbeddings()
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
        retriever = db.as_retriever()
        kwargs = {
            "functions": [
                {
                    "name": "sayHi",
                    "description": "A function to greate the user when he write his name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The user name"
                            }
                        },
                        "required": ["name"]
                    }
                }
            ],
        }
        llm = ChatOpenAI(
            temperature = 0,
            model_kwargs = kwargs
        )
        template = """
        Use the following pieces of context to answer the question at the end.
        You are a student assistant to help students apply to OKTamam System.
        Never say you are an AI model, always refer to yourself as a student assistant.
        You must ask the student from his name first and then call sayHi function at once.
        If you do not know the answer say I will call my manager and get back to you.
        If the student wants to register you should ask him for some data one by one in separate questions:
        - Name
        - Phone
        - Email Address
        After the student enters all this data say Your data is saved and our team will call you.

        {context}

        {history}
        Question: {question}
        Helpful Answer:"""

        prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)

        memory = ConversationBufferMemory(
            input_key="question", memory_key="history", return_messages=True)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )

        response = qa.run("What is the size of the Sun")
        print(response)

        st.session_state.conversation = qa
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "send" not in st.session_state:
        st.session_state.send = False

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
