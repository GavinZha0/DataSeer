#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : model.py
# @IDE            : PyCharm
# @desc           : AiModel

from fastapi import Depends
from core.dependencies import Paging, QueryParams


class ModelParams(QueryParams):
    def __init__(self, params: Paging = Depends()):
        super().__init__(params)