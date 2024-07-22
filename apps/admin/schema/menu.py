#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : menu.py
# @IDE            : PyCharm
# @desc           : pydantic model

from pydantic import BaseModel, Field, ConfigDict
from core.data_types import DatetimeStr


class Menu(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    pid: int | None = Field(None, title="Pid")
    name: str = Field(..., title="Name")
    title: str = Field(..., title="Title")
    icon: str | None = Field(None, title="Icon")
    pos: int = Field(..., title="Position")
    subreport: bool = Field(False, title="Is sub report")
    component: str | None = Field(None, title="Component")
    path: str | None = Field(None, title="Path")
    redirect: str | None = Field(None, title="Redirect")
    active: bool = Field(False, title="Is active")
    deleted: bool = Field(False, title="Is deleted")
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')


# 路由展示
class RouterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    title: str
    component: str | None = None
    path: str
    redirect: str | None = None
    order: int | None = None
    icon: str | None = None
    subreport: bool | None = False
    children: list[dict] = []