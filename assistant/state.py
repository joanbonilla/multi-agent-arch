import operator
from typing import Annotated, Any, Dict, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AppAgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages : With user question, error messages, reasoning
        query : the customer query
        cards : the results from the ground truth
        template : the template layout to render
    """

    messages: Annotated[list[AnyMessage], add_messages]
    sender: str
    results: Dict[str, Any]
    cards: Dict[str, Any]
