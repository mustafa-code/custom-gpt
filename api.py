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

def sayHi(data):
    return json.dumps({
        "status": True,
        "message": "Hi "+data["name"]+", enjoy our AI model O_o"
    })

def get_func_def():
    return [
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
    ]

def callChat(content, chat_id = None, type = "user", function = None):

    if chat_id is None:
        chat_id = str(uuid.uuid4())
        file_json = "histories/"+chat_id+'.json'
        messages = []
    else :
        file_json = "histories/"+chat_id+'.json'
        messages = json.load(open(file_json))

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
            source_documents.append({
                "source": doc.metadata["source"],
                "page_content": doc.page_content,
            })
            datasource += doc.page_content + "\n\n"

    messages.insert(0, {
        "role": "system", 
        "content": f"""
            You must ask the student from his name first and then call sayHi function at once.

            {datasource}
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

    if response.get("function_call"):
        available_functions = {
            "sayHi": sayHi,
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

    response = callChat(question, chat_id)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, port=5110)
