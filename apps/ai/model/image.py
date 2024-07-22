#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : image.py
# @IDE            : PyCharm
# @desc           : AI image

from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from apps.admin.model import SysOrg
from apps.ai.model.model import AiModel
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, ForeignKey, Text


class AiImage(BaseDbModel):
    __tablename__ = "ai_image"
    __table_args__ = ({'comment': 'AI image'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    group: Mapped[Optional[str]] = mapped_column(String(64), default='default', comment="Group")
    type: Mapped[str] = mapped_column(String(64), comment="Type")
    area: Mapped[Optional[str]] = mapped_column(String(64), comment="Area")
    platform: Mapped[Optional[str]] = mapped_column(String(64), default="DJL", comment="Platform")
    platform_ver: Mapped[Optional[str]] = mapped_column(String(16), comment="Platform version")
    content: Mapped[Optional[str]] = mapped_column(Text, comment="Content")
    public: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is public")

    # bind to unique model
    model_id: Mapped[int] = mapped_column(ForeignKey('ai_model.id'), comment="Model id")
    model: Mapped[AiModel] = relationship(lazy='selectin')

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')
