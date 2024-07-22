#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : flow.py
# @IDE            : PyCharm
# @desc           : ML flow

from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from apps.admin.model import SysOrg
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, Integer, ForeignKey, Text, DateTime


class Flow(BaseDbModel):
    __tablename__ = "ml_flow"
    __table_args__ = ({'comment': 'ML flow'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    group: Mapped[Optional[str]] = mapped_column(String(64), default='default', comment="Group")
    config: Mapped[Optional[str]] = mapped_column(Text, comment="Config")
    workflow: Mapped[Optional[str]] = mapped_column(Text, comment="Workflow")
    canvas: Mapped[Optional[str]] = mapped_column(Text, comment="Canvas")
    x6_ver: Mapped[Optional[str]] = mapped_column(String(8), comment="X6 version")
    version: Mapped[Optional[str]] = mapped_column(String(8), comment="Version")
    status: Mapped[Optional[str]] = mapped_column(String(16), comment="Status")
    error: Mapped[Optional[str]] = mapped_column(Text, comment="Error")
    public: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is public")

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')