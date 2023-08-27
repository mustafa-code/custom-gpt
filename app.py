import requests
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

def get_pdf_text(file_list):
    text = ""
    for file in file_list:
        try:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()+"\n"
        except PdfReadError:
            text += file.read().decode("utf-8")+"\n"
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def print_data():
    print("Data: here...........")

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature = 0)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    template = """

    You are a student assestant to help students appling in OKtamam System.
    If you do not know the answer say I will call my manager and get back to you and then call print_data function.
    If the student want to register you should ask him for some data one by one:
     - What is your Name
     - What is your Phone
     - What is your Email
    After the student enter all this data then say your data saved and then call print_data function.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)

    memory = ConversationBufferMemory(
        input_key="question", memory_key="history", return_messages=True)

    conversation_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    return conversation_chain


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
        # botTemp = botTemp.replace("{{SOURCE_DOCUMENT}}", message["source_documents"][0].page_content)
        st.write(botTemp, unsafe_allow_html=True)

        userTemp = user_template.replace("{{MSG}}", message["query"])
        st.write(userTemp, unsafe_allow_html=True)    

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=["pdf", "txt"])
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write("Chunks length: {number}".format(number=len(text_chunks)))

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
