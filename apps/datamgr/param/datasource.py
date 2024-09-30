#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : datasource.py
# @IDE            : PyCharm
# @desc           : Datasource
import json

import execjs
from fastapi import Depends, Body
from core.dependencies import Paging, QueryParams
from typing import Optional


class DatasourceParams(QueryParams):
    def __init__(self, params: Paging = Depends()):
        super().__init__(params)


class SrcTableParams(QueryParams):
    def __init__(self,
                 id: int = Body(..., title="Source Id"),
                 locked: Optional[bool] = Body(None, title="Show locked tables")):
        super().__init__()
        self.id = id
        self.locked = locked


class ExeSqlParams(QueryParams):
    def __init__(self,
                 id: int = Body(..., title="Source Id"),
                 sql: str = Body(..., title="Sql text"),
                 variable: Optional[list] = Body(None, title="Variable"),
                 limit: Optional[int] = Body(None, title="Limit")):
        super().__init__()
        self.id = id
        self.sql = sql
        self.variable = variable
        self.limit = limit


class DsourceCreateParams(object):
    def __init__(self,
                 id: Optional[int] = Body(None, title="Datasource id"),
                 name: Optional[str] = Body(..., title="Name"),
                 desc: Optional[str] = Body(None, title="Description"),
                 group: Optional[str] = Body(None, title="Group"),
                 type: str = Body(..., title="TYpe"),
                 url: str = Body(..., title="Url"),
                 params: Optional[str] = Body(None, Body="Url parameters"),
                 username: str = Body(..., title="Username"),
                 password: str = Body(..., title="Password"),
                 public: Optional[bool] = Body(False, title="Is public"),
                 locked: Optional[str] = Body(None, title="Lock tables")):
        super().__init__()
        self.id = id
        self.name = name
        self.desc = desc
        self.group = group
        self.type = type
        self.url = url
        # convert list to string
        self.params = json.dumps(params) if params else None
        self.username = username
        self.password = password
        self.public = public
        self.locked = locked


