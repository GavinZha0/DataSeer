#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : user.py
# @IDE            : PyCharm
# @desc           : Datasource

from typing import Optional
from sqlalchemy.orm import relationship, Mapped, mapped_column
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, ForeignKey, Text
from apps.admin.model.org import SysOrg


class Datasource(BaseDbModel):
    __tablename__ = "data_source"
    __table_args__ = ({'comment': 'Datasource'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    group: Mapped[Optional[str]] = mapped_column(String(64), default='default', comment="Group")
    type: Mapped[str] = mapped_column(String(16), comment="Type")
    url: Mapped[str] = mapped_column(String(255), comment="URL")
    params: Mapped[Optional[str]] = mapped_column(String(255), comment="Parameters")
    username: Mapped[str] = mapped_column(String(64), comment="Username")
    password: Mapped[str] = mapped_column(String(255), comment="Password")
    version: Mapped[Optional[str]] = mapped_column(String(64), comment="Version")
    public: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is public")
    locked_table: Mapped[Optional[str]] = mapped_column(Text, comment="Locked tables")

    # has many datasets as children
    #datasets: Mapped[List["Dataset"]] = relationship(back_populates="datasource")

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')

