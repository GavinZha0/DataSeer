#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @desc           : pydantic model

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from core.data_types import DatetimeStr


class Data(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=('model_',))

    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    group: str = Field(..., title="Group")
    modelId: int | None = Field(None, title="Model Id", alias='model_id')
    dataset: str | None = Field(None, title="Dataset")
    fields: str | None = Field(None, title="Fields")
    result: str | None = Field(None, title="Result")
    public: bool = Field(False, title="Is public")
    orgId: int = Field(..., title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')

class AiDataExecute(BaseModel):
    endpoint: str = Field(..., title="Endpoint")
    data: dict = Field(..., title="Data")