from functools import lru_cache
from typing import Any

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

from app.core.config import get_settings

MONGO_SERVER_SELECTION_TIMEOUT_MS = 5_000


@lru_cache(maxsize=1)
def get_mongo_client() -> AsyncMongoClient[Any]:
    settings = get_settings()
    return AsyncMongoClient(
        settings.mongo_uri,
        appname=settings.service_name,
        serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
    )


def get_database() -> AsyncDatabase[Any]:
    settings = get_settings()
    return get_mongo_client()[settings.mongo_database]


async def check_database_connection() -> bool:
    try:
        await get_mongo_client().admin.command("ping")
    except PyMongoError:
        return False
    return True


async def close_mongo_client() -> None:
    if get_mongo_client.cache_info().currsize == 0:
        return

    await get_mongo_client().close()
    get_mongo_client.cache_clear()
