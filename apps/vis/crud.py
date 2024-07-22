#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : crud.py
# @IDE            : PyCharm
# @desc           : Data Access Layer

from . import schema, model
from sqlalchemy.ext.asyncio import AsyncSession
from core.crud import DalBase


class DatasetDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(DatasetDal, self).__init__()
        self.db = db
        self.model = model.Dataset
        self.schema = schema.Dataset


class DataviewDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(DataviewDal, self).__init__()
        self.db = db
        self.model = model.Dataview
        self.schema = schema.Dataview


class DatareportDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(DatareportDal, self).__init__()
        self.db = db
        self.model = model.Datareport
        self.schema = schema.Datareport
