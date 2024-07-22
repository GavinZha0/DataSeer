#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : model.py
# @IDE            : PyCharm
# @desc           : AI model

from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from apps.admin.model import SysOrg
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, Integer, ForeignKey, Text


class AiModel(BaseDbModel):
    __tablename__ = "ai_model"
    __table_args__ = ({'comment': 'AI model'})

    sid: Mapped[Optional[int]] = mapped_column(Integer, comment="Sid")
    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    category: Mapped[str] = mapped_column(String(64), default='default', comment="Category")
    type: Mapped[str] = mapped_column(String(32), comment="Type")
    tags: Mapped[Optional[str]] = mapped_column(String(64), comment="Tags")
    version: Mapped[str] = mapped_column(String(16), comment="Version")
    network: Mapped[Optional[str]] = mapped_column(String(16), comment="Network")
    framework: Mapped[str] = mapped_column(String(16), comment="Framework")
    frame_ver: Mapped[Optional[str]] = mapped_column(String(16), comment="Framework version")
    trainset: Mapped[Optional[str]] = mapped_column(String(64), comment="Trainset")
    files: Mapped[str] = mapped_column(Text, comment="Files")
    input: Mapped[str] = mapped_column(Text, comment="Input")
    output: Mapped[str] = mapped_column(Text, comment="Output")
    eval: Mapped[Optional[str]] = mapped_column(Text, comment="Eval")
    score: Mapped[Optional[int]] = mapped_column(Integer, comment="Score")
    price: Mapped[Optional[str]] = mapped_column(String(16), comment="Price")
    detail: Mapped[Optional[str]] = mapped_column(Text, comment="Detail")
    weblink: Mapped[Optional[str]] = mapped_column(String(64), comment="Weblink")
    public: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is public")

    model_id: Mapped[int] = mapped_column(Integer, comment="Model id")

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')