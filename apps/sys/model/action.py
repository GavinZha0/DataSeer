#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : access.py
# @IDE            : PyCharm
# @desc           : user action

from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Integer, ForeignKey, func
from core.database import Base

class LogAction(Base):
    __tablename__ = "log_action"
    __table_args__ = ({'comment': 'Log action'})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    ts_utc: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    username: Mapped[str] = mapped_column(String(64))
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey('sys_user.id'))
    type: Mapped[Optional[str]] = mapped_column(String(64))
    url: Mapped[Optional[str]] = mapped_column(String(64))
    module: Mapped[Optional[str]] = mapped_column(String(64))
    method: Mapped[Optional[str]] = mapped_column(String(64))
    tid: Mapped[Optional[int]] = mapped_column(Integer)
    param: Mapped[Optional[str]] = mapped_column(Text)
    result: Mapped[Optional[str]] = mapped_column(String(255), server_default='ok')
