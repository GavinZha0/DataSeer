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



class ModelDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(ModelDal, self).__init__()
        self.db = db
        self.model = model.AiModel
        self.schema = schema.Model


class ImageDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(ImageDal, self).__init__()
        self.db = db
        self.model = model.AiImage
        self.schema = schema.Image
