from agent.settings import settings


def test_defaults() -> None:
    assert settings.project.api_version == "v1"
    assert settings.zhipu.model == "glm-5.1"
    assert settings.zhipu.api_base == "https://open.bigmodel.cn/api/paas/v4"
