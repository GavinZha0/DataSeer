#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @desc           : AI image

from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from apps.admin.model import SysOrg
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, ForeignKey, Text, Integer


class AiData(BaseDbModel):
    __tablename__ = "ai_data"
    __table_args__ = ({'comment': 'AI data'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    group: Mapped[Optional[str]] = mapped_column(String(64), default='default', comment="Group")
    model_id: Mapped[int] = mapped_column(Integer, comment="ModelId")
    dataset: Mapped[Optional[str]] = mapped_column(Text, comment="Dataset")
    fields: Mapped[Optional[str]] = mapped_column(Text, comment="Field")
    result: Mapped[Optional[str]] = mapped_column(Text, comment="Result")
    public: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is public")

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')
