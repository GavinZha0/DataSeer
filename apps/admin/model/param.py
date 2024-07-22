#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : param.py
# @IDE            : PyCharm
# @desc           : sys parameter

from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey
from apps.admin.model import SysOrg
from db.db_base import BaseDbModel


class SysParam(BaseDbModel):
    __tablename__ = "sys_param"
    __table_args__ = ({'comment': 'Sys parameter'})

    name: Mapped[str] = mapped_column(String(64))
    desc: Mapped[Optional[str]] = mapped_column(String(128))
    group: Mapped[Optional[str]] = mapped_column(String(64))
    module: Mapped[str] = mapped_column(String(64))
    type: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(String(255))
    previous: Mapped[Optional[str]] = mapped_column(String(255))

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'))
    org: Mapped[SysOrg] = relationship(lazy='selectin')

