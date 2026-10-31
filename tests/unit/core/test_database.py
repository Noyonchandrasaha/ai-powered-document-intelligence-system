import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.core.config import Settings
from app.core.database import (
    check_database_connection,
    close_mongo_client,
    get_database,
    get_mongo_client,
)
from pymongo.errors import PyMongoError


@pytest.fixture(autouse=True)
def clear_mongo_client_cache():
    get_mongo_client.cache_clear()
    yield
    get_mongo_client.cache_clear()


@pytest.mark.unit
def test_get_mongo_client_uses_settings():
    settings = Settings(
        mongo_uri="mongodb://mongo:27017",
        service_name="test-api",
    )

    with (
        patch("app.core.database.get_settings", return_value=settings),
        patch("app.core.database.AsyncMongoClient") as mongo_client,
    ):
        get_mongo_client()

    mongo_client.assert_called_once_with(
        "mongodb://mongo:27017",
        appname="test-api",
        serverSelectionTimeoutMS=5_000,
    )


@pytest.mark.unit
def test_get_mongo_client_reuses_cached_client():
    with patch("app.core.database.AsyncMongoClient") as mongo_client:
        first_client = get_mongo_client()
        second_client = get_mongo_client()

    assert first_client is second_client
    mongo_client.assert_called_once()


@pytest.mark.unit
def test_get_database_returns_configured_database():
    database = MagicMock()
    client = MagicMock()
    client.__getitem__.return_value = database
    settings = Settings(mongo_database="test_database")

    with (
        patch("app.core.database.get_settings", return_value=settings),
        patch("app.core.database.get_mongo_client", return_value=client),
    ):
        result = get_database()

    assert result is database
    client.__getitem__.assert_called_once_with("test_database")


@pytest.mark.unit
def test_check_database_connection_returns_true_when_ping_succeeds():
    client = MagicMock()
    client.admin.command = AsyncMock()

    with patch("app.core.database.get_mongo_client", return_value=client):
        assert asyncio.run(check_database_connection()) is True

    client.admin.command.assert_called_once_with("ping")


@pytest.mark.unit
def test_check_database_connection_returns_false_when_ping_fails():
    client = MagicMock()
    client.admin.command = AsyncMock(side_effect=PyMongoError("MongoDB is unavailable"))

    with patch("app.core.database.get_mongo_client", return_value=client):
        assert asyncio.run(check_database_connection()) is False


@pytest.mark.unit
def test_close_mongo_client_closes_cached_client():
    with patch("app.core.database.AsyncMongoClient") as mongo_client:
        client = get_mongo_client()
        client.close = AsyncMock()
        asyncio.run(close_mongo_client())

    client.close.assert_called_once()
    mongo_client.assert_called_once()
    assert get_mongo_client.cache_info().currsize == 0


@pytest.mark.unit
def test_close_mongo_client_does_nothing_without_cached_client():
    with patch("app.core.database.AsyncMongoClient") as mongo_client:
        asyncio.run(close_mongo_client())

    mongo_client.assert_not_called()
