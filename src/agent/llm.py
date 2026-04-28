from langchain_community.chat_models import ChatOpenAI, ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

from agent.settings import settings


def get_zhipu_chat_model() -> ChatZhipuAI:
    """Return a configured ChatZhipuAI instance."""
    return ChatZhipuAI(
        api_key=settings.zhipu.api_key.get_secret_value(),
        model=settings.zhipu.model,
    )


def get_embedding_model() -> ZhipuAIEmbeddings:
    """Return a configured Zhipu embeddings model."""
    return ZhipuAIEmbeddings(
        api_key=settings.zhipu.api_key.get_secret_value(),
        model="embedding-3",
    )


def get_qwen_local_model() -> ChatOpenAI:
    """Create a ChatOpenAI-compatible client pointed at Qwen's local API."""
    return ChatOpenAI(
        model=settings.qwen.model,
        api_key=settings.qwen.api_key.get_secret_value(),
        base_url=settings.qwen.api_base,
    )


def get_deepseek_chat_model() -> ChatDeepSeek:
    return ChatDeepSeek(
        api_key=settings.deepseek.api_key,
        model=settings.deepseek.model,
        extra_body={
            "thinking": {"type": "disabled"},
        },
    )


def get_chat_model() -> BaseChatModel:
    return get_deepseek_chat_model()
    # return get_qwen_local_model()
    # return get_zhipu_chat_model()
