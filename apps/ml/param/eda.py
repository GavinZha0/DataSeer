#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/14
# @File           : eda.py
# @IDE            : PyCharm
# @desc           : MlEda


from fastapi import Depends
from core.dependencies import QueryParams, Paging


class EdaParams(QueryParams):
    def __init__(self, params: Paging = Depends()):
        super().__init__(params)
