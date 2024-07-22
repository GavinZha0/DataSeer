#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : org.py
# @IDE            : PyCharm
# @desc           : organization

from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, Integer, DateTime, ForeignKey


class SysOrg(BaseDbModel):
    __tablename__ = "sys_org"
    __table_args__ = ({'comment': 'Organization'})

    pid: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("sys_org.id"), comment="Pid")
    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    logo: Mapped[Optional[str]] = mapped_column(String(255), comment="Logo")
    active: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is active")
    exp_date: Mapped[Optional[datetime]] = mapped_column(DateTime, comment="Expiration date")
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is deleted")
