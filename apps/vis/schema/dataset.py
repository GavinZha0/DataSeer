#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : dataset.py
# @IDE            : PyCharm
# @desc           : pydantic model

import json
from pydantic import BaseModel, Field, ConfigDict, field_validator
from core.data_types import DatetimeStr


class Dataset(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    group: str | None = Field("default", title="Group")
    variable: str | None = Field(None, title="Variables")
    query: str | None = Field(None, title="Query text")
    finalQuery: str | None = Field(None, title="Final query text", alias='final_query')
    error: str | None = Field(None, title="Error")
    field: str | None = Field(None, title="Fields")
    graph: str | None = Field(None, title="Graph")
    graphVer: str | None = Field(None, title="Graph version", alias='graph_ver')
    public: bool = Field(False, title="Is public")
    sourceId: int = Field(..., title="Datasource id", alias='source_id')
    orgId: int = Field(..., title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')

    @field_validator("variable", 'field', mode='after')
    @classmethod
    def config_validator(cls, v: str) -> dict:
        if v is not None and len(v) > 0:
            return json.loads(v)
            # return execjs.eval(v)
        else:
            return v