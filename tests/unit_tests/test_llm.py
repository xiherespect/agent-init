from agent.llm import get_chat_model


def test_get_chat_model() -> None:
    model = get_chat_model()
    result = model.invoke("Hello, world!")
    assert result is not None
