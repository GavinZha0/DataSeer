#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : role.py
# @IDE            : PyCharm
# @desc           : 角色模型

from typing import Optional
from sqlalchemy.orm import relationship, Mapped, mapped_column
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, ForeignKey
from .org import SysOrg
from .m2m import table_sys_role_menu_permit


class SysRole(BaseDbModel):
    __tablename__ = "sys_role"
    __table_args__ = ({'comment': 'Role'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    active: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is active")

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')
