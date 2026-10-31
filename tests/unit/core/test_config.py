import pytest
from app.core.config import Settings
from pydantic import ValidationError


@pytest.mark.unit
def test_api_prefix_is_normalized():
    settings = Settings(api_prefix="api/v2")

    assert settings.api_prefix == "/api/v2"


@pytest.mark.unit
def test_mongo_settings_are_configured():
    settings = Settings(
        mongo_uri="mongodb://mongo:27017",
        mongo_database="test_database",
    )

    assert settings.mongo_uri == "mongodb://mongo:27017"
    assert settings.mongo_database == "test_database"


@pytest.mark.unit
def test_openai_settings_are_configured_without_exposing_api_key():
    settings = Settings(
        openai_api_key="test-api-key",
        openai_model="gpt-test",
    )

    assert settings.openai_api_key == "test-api-key"
    assert settings.openai_model == "gpt-test"
    assert "test-api-key" not in repr(settings)


@pytest.mark.unit
def test_empty_api_prefix_is_rejected():
    with pytest.raises(ValidationError):
        Settings(api_prefix=" ")


@pytest.mark.unit
def test_production_rejects_wildcard_hosts():
    with pytest.raises(ValidationError):
        Settings(environment="production", allowed_hosts=["*"])
