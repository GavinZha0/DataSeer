#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : org.py
# @IDE            : PyCharm
# @desc           : SysOrg

from fastapi import Depends
from core.dependencies import Paging, QueryParams


class OrgParams(QueryParams):
    def __init__(self, params: Paging = Depends()):
        super().__init__(params)
