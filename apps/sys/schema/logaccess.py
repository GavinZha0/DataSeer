#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/19
# @File           : logaccess.py
# @IDE            : PyCharm
# @desc           : pydantic model

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class Logaccess(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    tsUtc: datetime = Field(..., title="None", alias='ts_utc')
    username: str = Field(..., title="None")
    userId: int | None = Field(None, title="None", alias='user_id')
    login: bool = Field(True, title="None")
    ip: str | None = Field(None, title="None")
    os: str | None = Field(None, title="None")
    browser: str | None = Field(None, title="None")
    lang: str | None = Field(None, title="None")
    timeZone: str | None = Field(None, title="None", alias='time_zone')
    location: str | None = Field(None, title="None")
    result: str | None = Field(None, title="None")
