#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : model.py
# @IDE            : PyCharm
# @desc           : pydantic model

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from core.data_types import DatetimeStr


class Model(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=('model_',))

    id: int = Field(..., title="Id")
    sid: int | None = Field(None, title="Sid")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    category: str = Field("default", title="Category")
    type: str = Field(..., title="Type")
    tags: str | None = Field(None, title="Tags")
    version: str = Field(..., title="Version")
    network: str | None = Field(None, title="Network")
    framework: str = Field(..., title="Framework")
    frameVer: str | None = Field(None, title="Framework version", alias='frame_ver')
    trainset: str | None = Field(None, title="Trainset")
    files: str = Field(..., title="Files")
    input: str = Field(..., title="Input")
    output: str = Field(..., title="Output")
    eval: str | None = Field(None, title="Eval")
    score: int | None = Field(None, title="Score")
    price: str | None = Field(None, title="Price")
    detail: str | None = Field(None, title="Detail")
    weblink: str | None = Field(None, title="Weblink")
    public: bool = Field(False, title="Is public")
    modelId: int | None = Field(..., title="Model id", alias='model_id')
    orgId: int = Field(..., title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')

