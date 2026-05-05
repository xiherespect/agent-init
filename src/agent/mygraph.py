from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str


def node_a(state: State):
    return {"text": state["text"] + "A"}


def node_b(state: State):
    return {"text": state["text"] + "B"}


graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)

graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")
graph.add_edge("node_b", END)

app = graph.compile()

result = app.invoke({"text": ""})
print(result)
