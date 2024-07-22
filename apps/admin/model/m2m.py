#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : m2m.py
# @IDE            : PyCharm
# @desc           : 关联中间表

from core.database import Base
from sqlalchemy import ForeignKey, Column, Table, Integer


table_sys_user_role = Table(
    "sys_user_role",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("sys_user.id", ondelete="CASCADE")),
    Column("role_id", Integer, ForeignKey("sys_role.id", ondelete="CASCADE")),
)


table_sys_role_menu_permit = Table(
    "sys_role_menu_permit",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("role_id", Integer, ForeignKey("sys_role.id", ondelete="CASCADE")),
    Column("menu_id", Integer, ForeignKey("sys_menu.id", ondelete="CASCADE")),
)
