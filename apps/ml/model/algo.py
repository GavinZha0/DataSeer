#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : algo.py
# @IDE            : PyCharm
# @desc           : ML algo

from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from apps.admin.model import SysOrg
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, ForeignKey, Text, Integer


class Algo(BaseDbModel):
    __tablename__ = "ml_algo"
    __table_args__ = ({'comment': 'ML algo'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    group: Mapped[Optional[str]] = mapped_column(String(64), default='default', comment="Group")
    category: Mapped[str] = mapped_column(String(16), comment="Category")
    framework: Mapped[Optional[str]] = mapped_column(String(16), default='python', comment="Framework")
    frame_ver: Mapped[Optional[str]] = mapped_column(String(8), default='3.11', comment="Frame version")
    algo_name: Mapped[Optional[str]] = mapped_column(String(64), comment="Algo name")
    src_code: Mapped[Optional[str]] = mapped_column(Text, comment="Content")
    data_cfg: Mapped[Optional[str]] = mapped_column(Text, comment="Attribute")
    train_cfg: Mapped[Optional[str]] = mapped_column(Text, comment="Config")
    public: Mapped[bool] = mapped_column(Boolean, default=False, comment="Public")

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')

