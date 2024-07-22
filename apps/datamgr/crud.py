#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : crud.py
# @IDE            : PyCharm
# @desc           : Data Access Layer
from . import schema, model
from core.crud import DalBase
from sqlalchemy.ext.asyncio import AsyncSession



class DatasourceDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(DatasourceDal, self).__init__()
        self.db = db
        self.model = model.Datasource
        self.schema = schema.Datasource
