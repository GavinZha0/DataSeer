#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/19
# @File           : logaction.py
# @IDE            : PyCharm
# @desc           : LogAction

from fastapi import Depends
from core.dependencies import Paging, QueryParams


class LogactionParams(QueryParams):
    def __init__(self, params: Paging = Depends()):
        super().__init__(params)
