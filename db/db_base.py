# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/4/18
# @File           : db_base.py
# @desc           : ORM model

from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from core.database import Base
from sqlalchemy import DateTime, Integer, func, inspect, String


class BaseDbModel(Base):
    __abstract__ = True

    id:  Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_by: Mapped[str | None] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )

    @classmethod
    def get_column_attrs(cls) -> list:
        mapper = inspect(cls)
        return mapper.column_attrs.keys()

    @classmethod
    def get_attrs(cls) -> list:
        mapper = inspect(cls)
        return mapper.attrs.keys()

    @classmethod
    def get_relationships_attrs(cls) -> list:
        mapper = inspect(cls)
        return mapper.relationships.keys()

