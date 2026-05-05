from typing import Annotated

from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agent.llm import get_chat_model


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def search(query: str) -> str:
    """Search information by query."""
    return f"搜索结果：{query} 的相关资料"


tools = [search]
tool_node = ToolNode(tools)


def agent_node(state: AgentState) -> dict[str, list[AnyMessage]]:
    """Call the model and let it decide whether to use tools."""
    llm_with_tools = get_chat_model().bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route to tools when the model produced tool calls."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"


graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "agent")

app = graph.compile(name="React Agent")
