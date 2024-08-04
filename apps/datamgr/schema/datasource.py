#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : datasource.py
# @IDE            : PyCharm
# @desc           : pydantic model
import json
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator
from core.data_types import DatetimeStr


class Datasource(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    group: str | None = Field("default", title="Group")
    type: str = Field(..., title="Type")
    url: str = Field(..., title="URL")
    params: str | None = Field(None, title="Parameters")
    username: str = Field(..., title="Username")
    password: str = Field(..., title="Password")
    version: str | None = Field(None, title="Version")
    public: bool = Field(False, title="Is public")
    lockedTable: str | None = Field(None, title="Locked tables", alias='locked_table')
    orgId: int = Field(..., title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')

    @field_validator('params', mode='after')
    @classmethod
    def config_validator(cls, v: dict) -> str:
        if v is not None:
            return json.dumps(v)
            # return execjs.eval(v)
        else:
            return v

class DatasourceSets(BaseModel):
    id: int = Field(..., title="Source Id"),
    type: str = Field(..., title="Source Type")