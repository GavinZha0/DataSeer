#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/14
# @File           : eda.py
# @IDE            : PyCharm
# @desc           : pydantic model

import json
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from core.data_types import DatetimeStr


class Eda(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    group: str | None = Field("default", title="Group")
    config: str | None = Field(None, title="Analysis config")
    public: bool = Field(False, title="Is public")
    datasetId: int = Field(..., title="Dataset id", alias='dataset_id')
    org_id: int = Field(..., title="Org id")
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')
    datasetName: Optional[str] = None

    @field_validator('config', mode='after')
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

    @model_validator(mode='before')
    @classmethod
    def validate_all(cls, values):
        if values.dataset is not None:
            # build extra field (calculated field) 'datasetName'
            values.datasetName = values.dataset.group + '/' + values.dataset.name
        return values


class EdaBuildParam(BaseModel):
    dataset_id: int = Field(..., title="Dataset id")
    tier: str = Field(..., title="Tier")
    kind: str = Field(..., title="Kind")
    config: dict = Field(..., title="Config")