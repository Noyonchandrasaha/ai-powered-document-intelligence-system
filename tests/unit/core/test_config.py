import pytest
from app.core.config import Settings
from pydantic import ValidationError


@pytest.mark.unit
def test_api_prefix_is_normalized():
    settings = Settings(api_prefix="api/v2")

    assert settings.api_prefix == "/api/v2"


@pytest.mark.unit
def test_empty_api_prefix_is_rejected():
    with pytest.raises(ValidationError):
        Settings(api_prefix=" ")


@pytest.mark.unit
def test_production_rejects_wildcard_hosts():
    with pytest.raises(ValidationError):
        Settings(environment="production", allowed_hosts=["*"])
