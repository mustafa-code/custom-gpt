import os
import json
import uuid
import openai
import datetime;
import requests;
from html_text import extract_text
from flask_cors import CORS, cross_origin
from flask import Flask, jsonify, request
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from consts import CHROMA_SETTINGS, PERSIST_DIRECTORY, SOURCE_DIRECTORY
from load_data import get_docs, split_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mysql.connector

openai.api_key = os.environ["OPENAI_API_KEY"]

app = Flask(__name__)
CORS(app)

def registerStudent(data):
    # Call an API to create student.
    otasUrl = os.environ['OTAS_URL']
    apiKey = os.environ['OTAS_API_KEY']

    response = requests.post(otasUrl+"api/v1/createLead", json = {
        "api_key": apiKey,
        "first_name": data["first_name"],
        "last_name": data["last_name"],
        "email": data["email"],
        "phone": data["phone"],
    })
    
    return json.dumps(response.json())

def askManager(data):
    print("askManager: "+json.dumps(data))
    return json.dumps({
        "status": True,
        "message": "Question have been sent to manager and he will continue the conversation."
    })

def get_func_def():
    return [
        {
            "name": "registerStudent",
            "description": "Get called when the user provieded lead info",
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {
                        "type": "string",
                        "description": "The student's first name"
                    },
                    "last_name": {
                        "type": "string",
                        "description": "The student's last name"
                    },
                    "phone": {
                        "type": "string",
                        "description": "The student's phone"
                    },
                    "email": {
                        "type": "string",
                        "description": "The student's email"
                    },
                    "lang": {
                        "type": "string",
                        "description": "The user's conversation language as a lang code like en, ar, or tr"
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "The chat id provided in System role."
                    },
                },
                "required": ["name", "phone", "email", "nationalityId", "residenceId", "programName", "universityName", "chat_id"]
            }
        },
        {
            "name": "askManager",
            "description": "Get called when the answer is not known to the model",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question"
                    },
                    "studentData": {
                        "type": "string",
                        "description": "The student's data in JSON format"
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "The chat id provided in System role."
                    },
                    "lang": {
                        "type": "string",
                        "description": "The conversation language as a lang code like en, ar, or tr"
                    },
                },
                "required": ["question", "chat_id"]
            }
        }
    ]

def save_history(chat_id, content, response, source_documents, timestamp):
    source_history_path = "histories/source_docs/"+chat_id+".json"
    if os.path.isfile(source_history_path):
        source_history = json.load(open(source_history_path))
    else :
        source_history = []
    end_at = datetime.datetime.now().timestamp()
    source_history.append({
        "request_date": timestamp,
        "question": content,
        "response": response,
        "source_documents": source_documents,
        "end_at": end_at,
    })
    source_file = open(source_history_path, "w")  
    json.dump(source_history, source_file)
    source_file.close()

def callChat(content, chat_id = None, type = "user", function = None):

    file_json = "histories/"+chat_id+'.json'
    if os.path.isfile(file_json):
        messages = json.load(open(file_json))
    else :
        messages = []

    timestamp = datetime.datetime.now().timestamp()

    datasource = ""
    source_documents = []
    if type == "user":
        embeddings = OpenAIEmbeddings()
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
        retriever = db.as_retriever()
        docs = retriever.get_relevant_documents(content)
        for doc in docs:
            if doc.page_content not in datasource:
                source_documents.append({
                    "source": doc.metadata["source"],
                    "page_content": doc.page_content,
                })
                datasource += doc.page_content + "\n\n"

    messages.insert(0, {
        "role": "system", 
        "content": f"""
            Use the following pieces of context to answer user question.
            You are a student assistant to help students apply to OKTamam System.
            Never say you are an AI model, always refer to yourself as a students assistant.
            If you do not know the answer call askManager Function and send the user question and the language of the conversation to it, with chat id.
            Always answer with shortes answer you can, do not say too much words.
            Never say reach out to the university directly or any other similar sentences, instead ask the manager and he will respond to you.
            It's important to append student language and chat id, when calling any functions.
            If the student wants to register you should ask him for some data one by one in separate questions:
            - First Name
            - Last Name
            - Phone
            - Email Address
            when the user give you his/her name, email, and phone number and add user language, and call the registerStudent Function.
            Never ask the student about his or her country IDs or show him or her any IDs.
            If there any issue occur then you must call askManager Function and send the question, the language of the conversation and all other paramters in JSON format caused the issue.
            if the information isn't exist in document or you don't know the answer never say no information, instead call the askManager function.

            Chat Id: '{chat_id}'
            Context: '{datasource}'
        """
    })

    # Create chat message last item in the conversation
    message = {
        "role": type, 
        "content": content,
    }
    if function is not None:
        message["name"] = function

    messages.append(message)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=get_func_def(),
        temperature=0,
    )["choices"][0]["message"]

    messages.append(response)

    # Save the new messages (conversation) in json file
    save_file = open(file_json, "w")  
    messages.pop(0)
    json.dump(messages, save_file)  
    save_file.close()  

    save_history(chat_id, content, response, source_documents, timestamp)

    if response.get("function_call"):
        available_functions = {
            "registerStudent": registerStudent,
            "askManager": askManager,
        }
        function_name = response["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response["function_call"]["arguments"])
        function_response = fuction_to_call(function_args)
        return callChat(function_response, chat_id, "function", function_name)

    source_history_path = "histories/source_docs/"+chat_id+".json"
    if os.path.isfile(source_history_path):
        source_history = json.load(open(source_history_path))
    else :
        source_history = []

    diff = source_history[(len(source_history) - 1)]["end_at"] - source_history[0]["request_date"]
    
    seconds = diff % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return {
        "response": response, 
        "chat_id": chat_id, 
        "message_id": (len(source_history) - 1),
        "conversation_duration": "%d:%02d:%02d" % (hour, minutes, seconds),
        "total_requests": len(source_history),
        # "source_documents": source_documents
    }

def scrape_and_save_embeddings(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Extract text content from the response
        text_content = extract_text(response.text)
        timestamp = datetime.datetime.now().timestamp()

        path = f"{SOURCE_DIRECTORY}/file-{timestamp}.txt"
        with open(path, 'w', encoding='utf-8') as file:
            file.write(text_content)

        documents = get_docs([path])
        text_documents = split_documents(documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(text_documents)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=CHROMA_SETTINGS,
        )
        db.persist()
        db = None


        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to scrape the content: {str(e)}")
        return False

@app.route("/api/prompt_route", methods=["POST"])
def prompt_route():
    data = request.get_json()

    question = data.get("question")
    chat_id = data.get("chat_id")
    # with_docs = request.form.get("with_docs")

    if chat_id is None:
        chat_id = str(uuid.uuid4())

    response = callChat(question, chat_id)

    return jsonify(response)

@app.route("/api/report_answer", methods=["POST"])
def report_answer():
    data = request.get_json()

    message_id = data.get("message_id")
    chat_id = data.get("chat_id")

    source_history_path = "histories/source_docs/"+chat_id+".json"
    if os.path.isfile(source_history_path):
        messages = json.load(open(source_history_path))

        diff = messages[(len(messages) - 1)]["end_at"] - messages[0]["request_date"]
        
        seconds = diff % (24 * 3600)
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60

        
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Admin123456",
            database="chat_db"
        )
        mycursor = mydb.cursor()

        sql = "INSERT INTO reports (question, answer, source_docs, duration, chat_id) VALUES (%s, %s, %s, %s, %s)"
        val = (
            messages[message_id]["question"], 
            messages[message_id]["response"]["content"], 
            json.dumps(messages[message_id]["source_documents"]), 
            "%d:%02d:%02d" % (hour, minutes, seconds),
            chat_id,
        )
        mycursor.execute(sql, val)

        mydb.commit()

        print(mycursor.rowcount, "record inserted.")

        return jsonify({
            "status": True,
            "affected_rows": mycursor.rowcount,
            "message": "Data saved successfully",
        })
    else :
        return jsonify({
            "status": False,
            "message": "Invalid chat id, or conversation has been deleted."
        })

@app.route("/api/load_url", methods=["POST"])
def load_url():
    data = request.get_json()

    url = data.get("url")
    status = scrape_and_save_embeddings(url)

    return jsonify({
        "status": status,
        "message": ("Url saved successfully" if status else "Fail to save URL")
    })

if __name__ == '__main__':
    app.run(debug=False, port=5110)
