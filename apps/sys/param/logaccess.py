#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/19
# @File           : logaccess.py
# @IDE            : PyCharm
# @desc           : LogAccess
import datetime

from fastapi import Depends
from core.dependencies import Paging, QueryParams


class LogaccessParams(QueryParams):
    def __init__(self, params: Paging = Depends()):
        super().__init__(params)