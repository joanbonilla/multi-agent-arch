from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Literal

from assistant.nodes import TOOLS, cards_node, layout_node
from assistant.state import AppAgentState
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolExecutor

tool_executor = ToolExecutor(TOOLS)

def router(state) -> Literal["call_tool", "__end__", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    if "Respuesta:" in last_message.content:
        return "__end__"
    return "continue"

def call_tool(state: AppAgentState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_invocations = []
    for tool_call in last_message.tool_calls:
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        tool_invocations.append(action)

    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    
    responses = tool_executor.batch(tool_invocations, return_exceptions=True)

    tool_messages = [
        ToolMessage(
            content=str(response),
            name=tc["name"],
            tool_call_id=tc["id"],
        )
        for tc, response in zip(last_message.tool_calls, responses)
    ]

    return {
        "messages": tool_messages,
        "results": responses
    }

flow = StateGraph(AppAgentState)

flow.add_node("assistant", cards_node)
flow.add_node("layout_generator", layout_node)
flow.add_node("call_tool", call_tool)

flow.add_conditional_edges(
    "assistant",
    router,
    {"continue": "layout_generator", "call_tool": "call_tool", "__end__": END},
)

flow.add_conditional_edges(
    "layout_generator",
    router,
    {"continue": "assistant", "call_tool": "call_tool", "__end__": END},
)

flow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "assistant": "assistant",
        "layout_generator": "layout_generator",
    },
)

flow.set_entry_point("assistant")

app_graph = flow.compile()

app_graph.get_graph().draw_mermaid_png(output_file_path="app_graph.png")
