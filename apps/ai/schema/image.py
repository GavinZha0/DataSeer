#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : image.py
# @IDE            : PyCharm
# @desc           : pydantic model

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from core.data_types import DatetimeStr


class Image(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=('model_',))

    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    group: str | None = Field("default", title="Group")
    type: str = Field(..., title="Type")
    area: str | None = Field(None, title="Area")
    platform: str | None = Field("DJL", title="Platform")
    platformVer: str | None = Field(None, title="Platform version", alias='platform_ver')
    content: str | None = Field(None, title="Content")
    public: bool = Field(False, title="Is public")
    modelId: int = Field(..., title="Model id", alias='model_id')
    orgId: int = Field(..., title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')

