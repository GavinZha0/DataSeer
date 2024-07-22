#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : algo.py
# @IDE            : PyCharm
# @desc           : pydantic model
import json
from pydantic import BaseModel, Field, ConfigDict, field_validator
from core.data_types import DatetimeStr


class Algo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    group: str | None = Field("default", title="Group")
    category: str = Field(..., title="Category")
    framework: str | None = Field("python", title="Framework")
    frameVer: str | None = Field("3.10", title="Frame version", alias='frame_ver')
    srcCode: str | None = Field(None, title="Source code", alias='src_code')
    attr: str | None = Field(None, title="Attribute")
    config: str | None = Field(None, title="Config")
    algoName: str | None = Field(None, title="Algo name", alias='algo_name')
    datasetId: int | None = Field(..., title="Dataset id", alias='dataset_id')
    public: bool = Field(False, title="Public")
    orgId: int = Field(..., title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')

    @field_validator('attr', 'config', mode='after')
    @classmethod
    def config_validator(cls, v: str) -> dict:
        if v is not None and len(v) > 0:
            return json.loads(v)
            # return execjs.eval(v)
        else:
            return v


class AlgoGetParam(BaseModel):
    framework: str = Field(..., title="Framework")
    category: str = Field(..., title="Category")


class AlgoExeParam(BaseModel):
    id: int = Field(..., title="Algo id")