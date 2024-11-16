#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @desc           : pydantic model

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from core.data_types import DatetimeStr


class Model(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=('model_',))

    id: int = Field(..., title="Id")
    name: str = Field(..., title="Name")
    desc: str | None = Field(None, title="Description")
    area: str = Field("data", title="Area")
    tags: str | None = Field(None, title="Tags")
    version: int = Field(..., title="Version")
    algoId: int | None = Field(None, title="Algo Id", alias='algo_id')
    runId: str | None = Field(None, title="Run Id", alias='run_id')
    rate: int | None = Field(None, title="Rate")
    price: str | None = Field(None, title="Price")
    deployTo: str | None = Field(None, title="Deploy To", alias='deploy_to')
    endpoint: str | None = Field(None, title="Endpoint")
    public: bool = Field(False, title="Is public")
    status: int = Field(0, title="0:idle; 1:serving; 2:exception; 3:unknown;")
    orgId: int = Field(..., title="Org id", alias='org_id')
    createdBy: str = Field(..., title="Created by", alias='created_by')
    createdAt: DatetimeStr = Field(..., title="Created at", alias='created_at')
    updatedBy: str | None = Field(..., title="Updated by", alias='updated_by')
    updatedAt: DatetimeStr | None = Field(..., title="Updated at", alias='updated_at')
    deployedBy: str | None = Field(..., title="Deployed by", alias='deployed_by')
    deployedAt: DatetimeStr | None = Field(..., title="Deployed at", alias='deployed_at')

class AiModelDeploy(BaseModel):
    id: int = Field(..., title="Store id")