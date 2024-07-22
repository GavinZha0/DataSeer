#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2023/7/16 12:42
# @File           : data_types.py
# @desc           : custom data type


import datetime
from typing import Annotated, Any
from bson import ObjectId
from pydantic import AfterValidator, PlainSerializer, WithJsonSchema
from .validator import *


def datetime_str_vali(value: str | datetime.datetime | int | float | dict):
    """
    Date time validation
    """
    if isinstance(value, str):
        pattern = "%Y-%m-%d %H:%M:%S"
        try:
            datetime.datetime.strptime(value, pattern)
            return value
        except ValueError:
            pass
    elif isinstance(value, datetime.datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(value, dict):
        # 用于处理 mongodb 日期时间数据类型
        date_str = value.get("$date")
        date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
        # 将字符串转换为datetime.datetime类型
        datetime_obj = datetime.datetime.strptime(date_str, date_format)
        # 将datetime.datetime对象转换为指定的字符串格式
        return datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
    raise ValueError("invalid data")


# 实现自定义一个日期时间字符串的数据类型
DatetimeStr = Annotated[
    str | datetime.datetime | int | float | dict,
    AfterValidator(datetime_str_vali),
    PlainSerializer(lambda x: x, return_type=str),
    WithJsonSchema({'type': 'string'}, mode='serialization')
]


# phone number
Telephone = Annotated[
    str,
    AfterValidator(lambda x: vali_telephone(x)),
    PlainSerializer(lambda x: x, return_type=str),
    WithJsonSchema({'type': 'string'}, mode='serialization')
]


# email
Email = Annotated[
    str,
    AfterValidator(lambda x: vali_email(x)),
    PlainSerializer(lambda x: x, return_type=str),
    WithJsonSchema({'type': 'string'}, mode='serialization')
]


def date_str_vali(value: str | datetime.date | int | float):
    """
    Date time validation
    """
    if isinstance(value, str):
        pattern = "%Y-%m-%d"
        try:
            datetime.datetime.strptime(value, pattern)
            return value
        except ValueError:
            pass
    elif isinstance(value, datetime.date):
        return value.strftime("%Y-%m-%d")
    raise ValueError("invalid data")


# Date string
DateStr = Annotated[
    str | datetime.date | int | float,
    AfterValidator(date_str_vali),
    PlainSerializer(lambda x: x, return_type=str),
    WithJsonSchema({'type': 'string'}, mode='serialization')
]


def object_id_str_vali(value: str | dict | ObjectId):
    """
    官方文档：https://docs.pydantic.dev/dev-v2/usage/types/datetime/
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value.get("$oid")
    elif isinstance(value, ObjectId):
        return str(value)
    raise ValueError("invalid ObjectId type")


ObjectIdStr = Annotated[
    Any,
    AfterValidator(object_id_str_vali),
    PlainSerializer(lambda x: x, return_type=str),
    WithJsonSchema({'type': 'string'}, mode='serialization')
]
