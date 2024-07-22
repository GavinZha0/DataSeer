#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/19
# @File           : logaction.py
# @IDE            : PyCharm
# @desc           : pydantic model

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class Logaction(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    tsUtc: datetime = Field(..., title="None", alias='ts_utc')
    username: str = Field(..., title="None")
    userId: int = Field(..., title="None", alias='user_id')
    type: str | None = Field(None, title="None")
    url: str | None = Field(None, title="None")
    module: str | None = Field(None, title="None")
    method: str | None = Field(None, title="None")
    tid: int | None = Field(None, title="None")
    param: str | None = Field(None, title="None")
    result: str | None = Field(None, title="None")

