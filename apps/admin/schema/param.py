#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : param.py
# @IDE            : PyCharm
# @desc           : pydantic model

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from core.data_types import DatetimeStr


class Param(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    name: str = Field(..., title="None")
    desc: str | None = Field(None, title="None")
    group: str | None = Field(None, title="None")
    module: str = Field(..., title="None")
    type: str = Field(..., title="None")
    value: str = Field(..., title="None")
    previous: str | None = Field(None, title="None")
    orgId: int = Field(..., title="None", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')

