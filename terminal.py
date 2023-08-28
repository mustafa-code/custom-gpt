import requests
import os
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
        # try:
        #     pdf_reader = PdfReader(file)
        #     for page in pdf_reader.pages:
        #         text += page.extract_text()+"\n"
        # except PdfReadError:
        #     text += file.read().decode("utf-8")+"\n"
        text += file.read()+"\n"
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
    Use the following pieces of context to answer the question at the end.
    You are a student assistant to help students apply to OKTamam System.
    Never say you are an AI model, always refer to yourself as a student assistant.
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

    conversation_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    return conversation_chain


def main():
    load_dotenv()
    # get pdf text

    files = os.listdir("source_documents")
    files_objects = []
    for file in files:
        file_object = open("source_documents/"+file, 'r')
        files_objects.append(file_object)

    # print(files_objects)

    raw_text = get_pdf_text(files_objects)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    conversation = get_conversation_chain(vectorstore)

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        response = conversation(query)
        answer = response["result"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

if __name__ == '__main__':
    main()
