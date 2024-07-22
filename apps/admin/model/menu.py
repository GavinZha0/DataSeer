#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : menu.py
# @IDE            : PyCharm
# @desc           : Menu

from typing import Optional
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column


class SysMenu(BaseDbModel):
    __tablename__ = "sys_menu"
    __table_args__ = ({'comment': 'Menu'})

    pid: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("sys_menu.id", ondelete='CASCADE'), comment="Pid")
    name: Mapped[str] = mapped_column(String(64), comment="Name")
    title: Mapped[str] = mapped_column(String(64), comment="Title")
    icon: Mapped[Optional[str]] = mapped_column(String(64), comment="Icon")
    pos: Mapped[int] = mapped_column(Integer, comment="Position")
    subreport: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is sub report")
    component: Mapped[Optional[str]] = mapped_column(String(64), comment="Component")
    path: Mapped[Optional[str]] = mapped_column(String(64), comment="Path")
    redirect: Mapped[Optional[str]] = mapped_column(String(64), comment="Redirect")
    active: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is active")
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is deleted")
