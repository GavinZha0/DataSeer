#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : user.py
# @IDE            : PyCharm
# @desc           : User

from typing import Optional
from datetime import datetime
from sqlalchemy.orm import relationship, Mapped, mapped_column
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, DateTime, ForeignKey
from passlib.context import CryptContext

from . import SysRole
from .org import SysOrg
from .m2m import table_sys_user_role

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


class SysUser(BaseDbModel):
    __tablename__ = "sys_user"
    __table_args__ = ({'comment': 'Users'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    password: Mapped[str] = mapped_column(String(128), comment="Password")
    realname: Mapped[str] = mapped_column(String(64), comment="Real name")
    desc: Mapped[Optional[str]] = mapped_column(String(64), comment="Description")
    email: Mapped[Optional[str]] = mapped_column(String(64), comment="Email")
    phone: Mapped[Optional[str]] = mapped_column(String(16), comment="Phone")
    avatar: Mapped[Optional[str]] = mapped_column(String(255), comment="Avatar")
    social: Mapped[Optional[str]] = mapped_column(String(255), comment="Social media")
    active: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is active")
    sms_code: Mapped[bool] = mapped_column(Boolean, default=False, comment="Sms verification")
    exp_date: Mapped[Optional[datetime]] = mapped_column(DateTime, comment="Expiration date")
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is deleted")

    roles: Mapped[set[SysRole]] = relationship(secondary=table_sys_user_role, lazy="selectin")

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), default=1, comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')


