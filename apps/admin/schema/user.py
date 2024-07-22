#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : user.py
# @IDE            : PyCharm
# @desc           : pydantic model

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from core.data_types import DatetimeStr


class User(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    password: str = Field(..., title="Password")
    realname: str = Field(..., title="Real name")
    desc: str | None = Field(None, title="Description")
    email: str | None = Field(None, title="Email")
    phone: str | None = Field(None, title="Phone")
    avatar: str | None = Field(None, title="Avatar")
    social: str | None = Field(None, title="Social media")
    active: bool = Field(False, title="Is active")
    sms_code: bool | None = Field(False, title="Sms verification")
    exp_date: datetime | None = Field(None, title="Expiration date")
    deleted: bool = Field(False, title="Is deleted")
    orgId: int = Field(1, title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')

