#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : access.py
# @IDE            : PyCharm
# @desc           : user access


import json
import datetime
from typing import Optional
from fastapi import Request
from starlette.requests import Request as StarletteRequest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column
from user_agents import parse
from config.settings import LOGIN_LOG_RECORD
from sqlalchemy import String, Boolean, DateTime, Integer, func
from core.database import Base


class LogAccess(Base):
    __tablename__ = "log_access"
    __table_args__ = ({'comment': 'Log access'})

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    ts_utc: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    username: Mapped[str] = mapped_column(String(64))
    user_id: Mapped[Optional[int]] = mapped_column(Integer)
    login: Mapped[bool] = mapped_column(Boolean,  default=True)
    ip: Mapped[Optional[str]] = mapped_column(String(64))
    os: Mapped[Optional[str]] = mapped_column(String(64))
    browser: Mapped[Optional[str]] = mapped_column(String(16))
    lang: Mapped[Optional[str]] = mapped_column(String(16))
    time_zone: Mapped[Optional[str]] = mapped_column(String(64))
    location: Mapped[Optional[str]] = mapped_column(String(64))
    result: Mapped[Optional[str]] = mapped_column(String(255), server_default='ok')

    @classmethod
    async def create_login_record(
            cls,
            db: AsyncSession,
            uid: int,
            username: str,
            login: bool,
            req: Request | StarletteRequest,
            resp: dict
    ):
        """
        创建登录记录
        :return:
        """
        if not LOGIN_LOG_RECORD:
            return None
        header = {}
        for k, v in req.headers.items():
            header[k] = v
        if isinstance(req, StarletteRequest):
            form = (await req.form()).multi_items()
            params = json.dumps({"form": form, "headers": header})
        else:
            body = json.loads((await req.body()).decode())
            params = json.dumps({"body": body, "headers": header})
        user_agent = parse(req.headers.get("user-agent"))
        system = f"{user_agent.os.family} {user_agent.os.version_string}"
        browser = f"{user_agent.browser.family} {user_agent.browser.version_string}"
        # ip = IPManage(req.client.host)
        # location = await ip.parse()

        obj = LogAccess(
            ts_utc=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            username=username,
            user_id=uid,
            login=login,
            ip=req.client.host,
            browser=browser,
            os=system,
            time_zone='',
            lang='en-US',
            location='',
            result='ok'
        )
        db.add(obj)
        await db.flush()