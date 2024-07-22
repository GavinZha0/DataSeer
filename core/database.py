# -*- coding: utf-8 -*-
# @version        : 1.0
# @Update Time    : 2024/4/18
# @File           : database.py
# @desc           : SQLAlchemy


from typing import AsyncGenerator
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker, AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, declared_attr
from config.settings import SQLALCHEMY_DATABASE_URL, REDIS_ENABLE, MONGO_DB_ENABLE
from fastapi import Request
from core.exception import CustomException
from motor.motor_asyncio import AsyncIOMotorDatabase

# Create db engine
async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=False,
    echo_pool=False,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=5,
    max_overflow=5,
    connect_args={}
)

# create db session
session_factory = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=async_engine,
    expire_on_commit=True,
    class_=AsyncSession
)


class Base(AsyncAttrs, DeclarativeBase):
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """
        将表名改为小写
        如果有自定义表名就取自定义，没有就取小写类名
        """
        table_name = cls.__tablename__
        if not table_name:
            model_name = cls.__name__
            ls = []
            for index, char in enumerate(model_name):
                if char.isupper() and index != 0:
                    ls.append("_")
                ls.append(char)
            table_name = "".join(ls).lower()
        return table_name


async def db_getter() -> AsyncGenerator[AsyncSession, None]:
    """
    get db session
    """
    async with session_factory() as session:
        async with session.begin():
            yield session


def redis_getter(request: Request) -> Redis:
    """
    get redis
    """
    if not REDIS_ENABLE:
        raise CustomException("Redis is unavailable！", desc="Enable settings.py: REDIS_ENABLE")
    return request.app.state.redis


def mongo_getter(request: Request) -> AsyncIOMotorDatabase:
    """
    get mongoDb
    """
    if not MONGO_DB_ENABLE:
        raise CustomException(
            msg="MongoDb is unavailable",
            desc="Enable settings.py: MONGO_DB_ENABLE"
        )
    return request.app.state.mongo
