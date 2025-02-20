#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : crud.py
# @IDE            : PyCharm
# @desc           : Data Access Layer

from . import model, schema
from core.crud import DalBase
from sqlalchemy.ext.asyncio import AsyncSession


class DatasetDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(DatasetDal, self).__init__()
        self.db = db
        self.model = model.Dataset
        self.schema = schema.Dataset


class EdaDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(EdaDal, self).__init__()
        self.db = db
        self.model = model.Eda
        self.schema = schema.Eda


class AlgoDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(AlgoDal, self).__init__()
        self.db = db
        self.model = model.Algo
        self.schema = schema.Algo


class FlowDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(FlowDal, self).__init__()
        self.db = db
        self.model = model.Workflow
        self.schema = schema.Workflow




