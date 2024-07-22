#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : dataview.py
# @IDE            : PyCharm
# @desc           : Dataview

from fastapi import Depends
from core.dependencies import Paging, QueryParams


class DataviewParams(QueryParams):
    def __init__(self, params: Paging = Depends()):
        super().__init__(params)
