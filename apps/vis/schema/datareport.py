#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : datareport.py
# @IDE            : PyCharm
# @desc           : pydantic model
import json

from pydantic import BaseModel, Field, ConfigDict, field_validator
from core.data_types import DatetimeStr


class Datareport(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    group: str | None = Field("default", title="Group")
    type: str = Field(..., title="Type")
    pages: str = Field(..., title="Pages")
    public: bool = Field(False, title="Is public")
    pubPub: bool = Field(True, title="Is public publish", alias='pub_pub')
    viewIds: str = Field(..., title="View ids", alias='view_ids')
    menuId: int | None = Field(..., title="Menu id", alias='menu_id')
    orgId: int = Field(..., title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')

    @field_validator("pages", 'viewIds', mode='after')
    @classmethod
    def config_validator(cls, v: str) -> dict:
        if v is not None and len(v) > 0:
            return json.loads(v)
            # return execjs.eval(v)
        else:
            return v
