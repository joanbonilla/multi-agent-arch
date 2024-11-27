import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum

from dotenv import load_dotenv
from langchain.output_parsers import EnumOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.function import FunctionMessage
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI, VertexAI


from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.json_chat.prompt import TEMPLATE_TOOL_RESPONSE
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.agents.format_scratchpad import format_log_to_messages

from assistant.state import AppAgentState
from db_setup import db

from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    AIMessage
)
import functools

load_dotenv()

llm = ChatVertexAI(model_name="gemini-1.5-flash-001", location="us-central1")


def create_agent(llm, tools, system_message: str):

    print("create_agent node...")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu tarea es asistir con el proceso de preguntas y dudas de tarjetas de crédito y débito. "
                "No intentes inferir una respuesta si no viene en el contexto la información necesaria. "
                "Usa las herramientas disponibles para responder la duda o necesidad del cliente y responde con la plantilla del layout de la pantalla. "
                "Si no eres capaz de responder completamente está bien, otro asistente podrá ayudar con diferentes tools donde tu no puedas. "
                "Si tu o otro asistente ha finalizado ha respondido la pregunta añade el prefijo 'Respuesta:' para que el equipo sepa cuando parar. "
                "Think step by step. "
                "Tienes acceso a las siguientes tools: {tool_names}.\n{system_message}"
                ,
            ),
            ("placeholder", "{messages}"),
        ]
    )

    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


@tool
def get_cards_catalog() -> Dict[str, Any]:
    """
    This function use a database to get all the available cards

    :return: all the cards retrieved from the database
    """

    print("get_cards_catalog...")

    documents = db.collection("Cards").get()
    cards = []
    for document in documents:
        cards.append(document.to_dict())
    
    return {
        "cards": cards
    }


@tool
def get_template_layout() -> Dict[str, Any]:
    """
    This function return the layout template for the frontend app

    :return: json describing the layout components for react app with a template
    """

    print("get_template_layout ...")


    return {
        "template_layout": "",
        "action": "rendering"
    }


TOOLS = [get_cards_catalog, get_template_layout]

def agent_node(state, agent, name):
    result = agent.invoke(state)
    
    print("--- State -----")
    print(state)
    print("--- Result -----")
    print(result)
    print("--------")
    
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)

    return {
        "messages": [result],
        "sender": name,
    }


cards_agent = create_agent(
    llm,
    [get_cards_catalog],
    system_message="Deberías recuperar la información de las tarjetas que se necesiten y responder las dudas del cliente. Si sabes la respuesta responde con el prefijo 'Respuesta:'",
)
cards_node = functools.partial(agent_node, agent=cards_agent, name="assistant")


layout_agent = create_agent(
    llm,
    [get_template_layout],
    system_message="Debes devolver la información del layout en la respuesta.",
)
layout_node = functools.partial(agent_node, agent=layout_agent, name="layout_generator")


