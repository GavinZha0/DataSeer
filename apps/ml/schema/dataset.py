#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/14
# @File           : dataset.py
# @IDE            : PyCharm
# @desc           : pydantic model

import json
import re
from typing import Optional

import execjs
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, computed_field
from core.data_types import DatetimeStr


class Dataset(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    desc: Optional[str] = Field(None, title="Description")
    group: Optional[str] = Field("default", title="Group")
    type: str = Field(..., title="type")
    variable: Optional[str] = Field(None, title="Variable")
    content: Optional[str] = Field(None, title="Sql query or file name")
    final_query: Optional[str] = Field(None, title="Final query")
    fields: str = Field(..., title="Field info")
    transform: Optional[str] = Field(None, title="Transform")
    f_count: Optional[int] = Field(None, title="Feature count")
    target: Optional[str] = Field(None, title="Target field")
    public: bool = Field(False, title="Is public")
    sourceId: int = Field(..., title="Datasource id", alias='source_id')
    orgId: int = Field(..., title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr = Field(..., title="Updated at", alias='updated_at')
    sourceName: Optional[str] = None


    @field_validator('variable', 'transform', 'fields', mode='after')
    @classmethod
    def fields_validator(cls, v: str) -> list:
        if v is not None and len(v) > 0:
            return json.loads(v)
            # convert string to json
            # execjs.eval can handle the json key without double quotation
            # execjs.eval can convert string to list(array)
            # return execjs.eval(v)
        else:
            return v

    @field_validator('target', mode='after')
    @classmethod
    def target_validator(cls, v: str) -> list:
        if v is not None and len(v) > 0:
            # list-like string without quotes
            # ex: [aaa,bbb,ccc]
            vv = re.sub(r'[\[\]]', '', v).split(',')
            vv = [name.strip() for name in vv]
            return vv
        else:
            return v


    @model_validator(mode='before')
    @classmethod
    def validate_all(cls, fields):
        if fields.datasource is not None:
            # build extra field (calculated field) 'sourceName'
            fields.sourceName = fields.datasource.group + '/' + fields.datasource.name
        return fields


    '''
    # AttributeError: 'Dataset' object has no attribute 'datasource'
    @computed_field()
    @property
    def sourceName(self) -> str|None:
        if self.datasource is not None:
            # build extra field (calculated field) 'sourceName'
            return self.datasource.group + '/' + self.datasource.name
        else:
            return None
    '''

class DatasetGetOne(BaseModel):
    id: int = Field(..., title="Dataset id")


class DatasetCreate(BaseModel):
    name: Optional[str] = Field(..., title="Name"),
    desc: Optional[str] = Field(None, title="Description"),
    group: Optional[str] = Field(None, title="Group"),
    type: str = Field(None, title="Type"),
    variable: Optional[list] = Field(None, title="Variables"),
    content: Optional[str] = Field(None, title="Sql query or file name"),
    fields: list = Field(..., Body="Field info"),
    transform: Optional[list] = Field(None, title="Transform"),
    target: Optional[list] = Field(None, title="target field"),
    public: Optional[bool] = Field(False, title="Is public"),
    sourceId: int = Field(..., title="Datasource id", alias='source_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')


    @field_validator('variable', 'transform', 'fields', 'target', mode='after')
    @classmethod
    def fields_validator(cls, v: dict) -> str:
        if v is not None:
            return json.dumps(v)
            # return execjs.eval(v)
        else:
            return v


class DatasetGetStat(BaseModel):
    id: int = Field(..., title="Source Id"),
    content: str = Field(..., title="Content"),
    type: str = Field(..., title="Type"),
    variable: Optional[list] = Field(None, title="Variable"),
    limit: Optional[int] = Field(None, title="Limit")

