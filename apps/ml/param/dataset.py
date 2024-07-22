#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/14
# @File           : dataset.py
# @IDE            : PyCharm
# @desc           : MlDataset

from fastapi import Depends
from core.dependencies import Paging, QueryParams



class DatasetParams(QueryParams):
    def __init__(self, params: Paging = Depends()):
        super().__init__(params)