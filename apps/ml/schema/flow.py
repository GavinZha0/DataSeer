#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : flow.py
# @IDE            : PyCharm
# @desc           : pydantic model

import json
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator
from core.data_types import DatetimeStr


class Flow(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    group: str | None = Field("default", title="Group")
    config: str | None = Field(None, title="Config")
    workflow: str | None = Field(None, title="Workflow")
    canvas: str | None = Field(None, title="Canvas")
    x6Ver: str | None = Field(None, title="X6 version", alias='x6_ver')
    version: str | None = Field(None, title="Version")
    public: bool = Field(False, title="Is public")
    orgId: int = Field(..., title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')

    @field_validator('config', 'workflow', 'canvas', mode='after')
    @classmethod
    def config_validator(cls, v: str) -> list:
        if v is not None and len(v) > 0:
            return json.loads(v)
            # convert string to json
            # execjs.eval can handle the json key without double quotation
            # execjs.eval can convert string to list(array)
            # return execjs.eval(v)
        else:
            return v