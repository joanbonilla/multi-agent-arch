import base64
import datetime
import json
import firebase_admin


from typing import Dict, List

#from dotenv import load_dotenv
from firebase_admin import credentials, firestore
from flask import Flask, abort, jsonify, render_template, request
from google.cloud.firestore_v1 import FieldFilter
from google.cloud.firestore_v1.field_path import FieldPath
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_firestore import FirestoreChatMessageHistory

from assistant.graph import app_graph

from db_setup import db
from flask_cors import CORS

#load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/assistant", methods=["GET"])
def assistant():
    query = request.args.get("query")
    thread_id = request.args.get("thread_id")

    response = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "user_id": "emKszv8xjISy446FJNmK",
            "query": query,
        }
    )

    chat_history = FirestoreChatMessageHistory(session_id=thread_id)
    chat_history.add_message(HumanMessage(content=query))
    chat_history.add_message(response["messages"][-1])

    print("***** App RESPONSE *****")
    
    print(response["messages"][-1])
    
    history = [
        {
            "role": (type(message).__name__),
            "content": message.content,
            "time": datetime.datetime.now().strftime("%H:%M"),
        }
        for message in chat_history.messages
        if message.content and type(message).__name__ != "ToolMessage"
    ]

    return {"messages": [history[-1]], "query": query}


@app.route("/chat", methods=["GET"])
def chat():
    graph_res = {}
    query = request.args.get("query")
    return compiled_graph.invoke({"query": query})


@app.route("/app", methods=["GET"])
def help():

    query = request.args.get("query")
    response = app_graph.invoke({
        "messages": [HumanMessage(content=query)],
        "query": query
        })

    print("***** App RESPONSE *****")
    print(response)
    message = response["messages"][-1]
    results = response["results"][0]
    
    return {
        "messages": [
            {"role": "llm", "text": message.content},
            {"role": "user", "text": query},
            ],
        "layout": results
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=8080)
