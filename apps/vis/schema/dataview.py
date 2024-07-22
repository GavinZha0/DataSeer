#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : dataview.py
# @IDE            : PyCharm
# @desc           : pydantic model

import json
from pydantic import BaseModel, Field, ConfigDict, field_validator
from core.data_types import DatetimeStr


class Dataview(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    group: str | None = Field("default", title="Group")
    type: str = Field(..., title="Type")
    dim: str = Field(..., title="Dimension")
    relation: str | None = Field(None, title="Relation")
    location: str | None = Field(None, title="Location")
    metrics: str = Field(..., title="Metrics")
    agg: str | None = Field(None, title="Aggression")
    prec: int | None = Field(None, title="Precision")
    filter: str | None = Field(None, title="Filter")
    sorter: str | None = Field(None, title="Sorter")
    variable: str | None = Field(None, title="Variable")
    calculation: str | None = Field(None, title="Calculation")
    model: str | None = Field(None, title="Model")
    libName: str = Field(..., title="Lib name", alias='lib_name')
    libVer: str | None = Field(None, title="Lib version", alias='lib_ver')
    libCfg: str | None = Field(None, title="Lib config", alias='lib_cfg')
    public: bool = Field(False, title="Is public")
    datasetId: int = Field(..., title="Dataset id", alias='dataset_id')
    orgId: int = Field(..., title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')

    @field_validator("dim", 'relation', 'location', 'metrics', 'filter', 'sorter', 'variable', 'calculation', 'model', 'libCfg', mode='after')
    @classmethod
    def config_validator(cls, v: str) -> dict:
        if v is not None and len(v) > 0:
            return json.loads(v)
            # return execjs.eval(v)
        else:
            return v
