#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : org.py
# @IDE            : PyCharm
# @desc           : pydantic model

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from core.data_types import DatetimeStr


class Org(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    pid: int | None = Field(None, title="Pid")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    logo: str | None = Field(None, title="Logo")
    active: bool = Field(False, title="Is active")
    expDate: datetime | None = Field(None, title="Expiration date", alias='exp_date')
    deleted: bool = Field(False, title="Is deleted")
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')
