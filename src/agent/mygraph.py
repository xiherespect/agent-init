from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    user_input: str
    route: str
    answer: str


def router_node(state: State):
    text = state["user_input"]

    if "资料" in text or "文档" in text or "PDF" in text:
        route = "rag"
    elif "写" in text or "润色" in text or "改写" in text:
        route = "write"
    else:
        route = "chat"

    return {"route": route}


def route_fn(state: State):
    return state["route"]


def rag_node(state: State):
    return {"answer": f"这是 RAG 路线：我会根据资料回答：{state['user_input']}"}


def write_node(state: State):
    return {"answer": f"这是写作路线：我会帮你润色或生成文本：{state['user_input']}"}


def chat_node(state: State):
    return {"answer": f"这是聊天路线：我直接回答：{state['user_input']}"}


graph = StateGraph(State)

graph.add_node("router", router_node)
graph.add_node("rag_node", rag_node)
graph.add_node("write_node", write_node)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    route_fn,
    {
        "rag": "rag_node",
        "write": "write_node",
        "chat": "chat_node",
    },
)

graph.add_edge("rag_node", END)
graph.add_edge("write_node", END)
graph.add_edge("chat_node", END)

app = graph.compile()
