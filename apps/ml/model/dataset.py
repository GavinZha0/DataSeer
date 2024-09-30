#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : dataset.py
# @IDE            : PyCharm
# @desc           : ML Dataset


from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from apps.admin.model import SysOrg
from apps.datamgr.model import Datasource
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, Integer, ForeignKey, Text


class Dataset(BaseDbModel):
    __tablename__ = "ml_dataset"
    __table_args__ = ({'comment': 'ML Dataset'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    group: Mapped[Optional[str]] = mapped_column(String(64), default='default', comment="Group")
    variable: Mapped[Optional[str]] = mapped_column(Text, comment="Variables")
    content: Mapped[Optional[str]] = mapped_column(Text, comment="Sql query or file name")
    final_query: Mapped[Optional[str]] = mapped_column(Text, comment="Query text")
    fields: Mapped[str] = mapped_column(Text, comment="Field info")
    transform: Mapped[Optional[str]] = mapped_column(Text, comment="Transform")
    f_count: Mapped[Optional[int]] = mapped_column(Integer, comment="feature count")
    target: Mapped[Optional[str]] = mapped_column(Text, comment="target field")
    public: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is public")

    # bind to unique datasource
    source_id: Mapped[int] = mapped_column(Integer, ForeignKey('data_source.id'), comment="Datasource id")
    datasource: Mapped[Datasource] = relationship(lazy='selectin')

    # has many dataviews as children
    # dataviews: Mapped[List["Dataview"]] = relationship(back_populates="dataset")

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')

