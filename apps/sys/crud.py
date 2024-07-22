#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2021/10/18 22:18
# @File           : crud.py
# @IDE            : PyCharm
# @desc           : 数据库 增删改查操作

from sqlalchemy.orm import joinedload
from . import model, schema
from sqlalchemy import select
from core.crud import DalBase
from sqlalchemy.ext.asyncio import AsyncSession

class DictTypeDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(DictTypeDal, self).__init__()
        self.db = db
        self.model = model.VadminDictType
        self.schema = schema.DictTypeSimpleOut

    async def get_dicts_details(self, dict_types: list[str]) -> dict:
        """
        获取多个字典类型下的字典元素列表
        """
        data = {}
        options = [joinedload(self.model.details)]
        objs = await DictTypeDal(self.db).get_datas(
            limit=0,
            v_return_objs=True,
            v_options=options,
            dict_type=("in", dict_types)
        )
        for obj in objs:
            if not obj:
                data[obj.dict_type] = []
                continue
            else:
                data[obj.dict_type] = [schema.DictDetailsSimpleOut.model_validate(i).model_dump() for i in obj.details]
        return data

    async def get_select_datas(self) -> list:
        """获取选择数据，全部数据"""
        sql = select(self.model)
        queryset = await self.db.execute(sql)
        return [schema.DictTypeOptionsOut.model_validate(i).model_dump() for i in queryset.scalars().all()]



class LogactionDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(LogactionDal, self).__init__()
        self.db = db
        self.model = model.LogAction
        self.schema = schema.Logaction


class LogaccessDal(DalBase):

    def __init__(self, db: AsyncSession):
        super(LogaccessDal, self).__init__()
        self.db = db
        self.model = model.LogAccess
        self.schema = schema.Logaccess
