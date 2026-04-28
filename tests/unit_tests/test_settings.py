from pathlib import Path

import pytest
from pydantic import SecretStr

from agent.settings import settings


def test_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    assert settings.project.api_version == "v1"
    assert settings.zhipu.model == "glm-5.1"
    assert settings.zhipu.api_base == "https://open.bigmodel.cn/api/paas/v4"
