import os
import json
import uuid
import openai
from flask import Flask, jsonify, request
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from consts import CHROMA_SETTINGS, PERSIST_DIRECTORY

openai.api_key = os.environ["OPENAI_API_KEY"]

app = Flask(__name__)

def registerStudent(data):
    print("registerStudent: "+json.dumps(data))
    return json.dumps({
        "status": False,
        "message": "Fail to save student data."
    })

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
                    "name": {
                        "type": "string",
                        "description": "The user's name"
                    },
                    "phone": {
                        "type": "string",
                        "description": "The user's phone"
                    },
                    "email": {
                        "type": "string",
                        "description": "The user's email"
                    },
                    "nationalityId": {
                        "type": "string",
                        "description": "The student's country nationality id"
                    },
                    "residenceId": {
                        "type": "string",
                        "description": "The student's residence country id"
                    },
                    "programName": {
                        "type": "string",
                        "description": "The program's name student want to apply in"
                    },
                    "universityName": {
                        "type": "string",
                        "description": "The school's name student want to apply in"
                    },
                    "fatherName": {
                        "type": "string",
                        "description": "The student's father name"
                    },
                    "motherName": {
                        "type": "string",
                        "description": "The student's mother name"
                    },
                    "lang": {
                        "type": "string",
                        "description": "The user's conversation language as a lang code like en, ar, or tr"
                    },
                },
                "required": ["name", "phone", "email", "nationalityId", "residenceId", "programName", "universityName"]
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
                    "lang": {
                        "type": "string",
                        "description": "The conversation language as a lang code like en, ar, or tr"
                    },
                },
                "required": ["question"]
            }
        }
    ]

def save_history(chat_id, content, response, source_documents):
    source_history_path = "histories/source_docs/"+chat_id+".json"
    if os.path.isfile(source_history_path):
        source_history = json.load(open(source_history_path))
    else :
        source_history = []
    source_history.append({
        "question": content,
        "response": response,
        "source_documents": source_documents,
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
            You should answer only to the request and questions related to (learning,universities,Oktamam company), if so apolgaize to the user.
            Never say you are an AI model, always refer to yourself as a student assistant.
            If you do not know the answer call askManager Function and send the user question and the language of the conversation to it.
            Always answer with shortes answer you can, do not say too much.
            Never say reach out to the university directly or any other similar sentences, instead ask the manager and he will respond to you.
            It's important to append student language, when calling any functions.
            If the student wants to register you should ask him for some data one by one in separate questions:
            - Name
            - Phone
            - Email Address
            - Nationality
            - Residence Country
            - Father name
            - Mother name
            when the user give you his/her name, email, and phone number and add user language, program name and university name to the parameters, for Nationality and Residence Country you have to match with countries ids and use the id of the country as a parameter instead of the name and call the registerStudent Function.
            If there any issue occur then you must call askManager Function and send the question, the language of the conversation and all other paramters in JSON format caused the issue.


            Context: '{datasource}'
            Country ids:
            - Sudan: 121
            - Egypt: 122
            - Turkey: 123
            - United Arab Emirates: 124
            - Saudi Arabia: 125
            - Qatar: 126
            - Oman: 127
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
        functions=get_func_def()
    )["choices"][0]["message"]

    messages.append(response)

    # Save the new messages (conversation) in json file
    save_file = open(file_json, "w")  
    messages.pop(0)
    json.dump(messages, save_file)  
    save_file.close()  

    save_history(chat_id, content, response, source_documents)

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

    return {
        "response": response, 
        "chat_id": chat_id, 
        # "source_documents": source_documents
    }

@app.route("/api/prompt_route", methods=["POST"])
def prompt_route():
    question = request.form.get("question")
    chat_id = request.form.get("chat_id")
    with_docs = request.form.get("with_docs")

    if chat_id is None:
        chat_id = str(uuid.uuid4())

    response = callChat(question, chat_id)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, port=5110)
