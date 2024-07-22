#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : eda.py
# @IDE            : PyCharm
# @desc           : ML Eda


from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from apps.admin.model import SysOrg
from apps.ml.model import Dataset
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, Integer, ForeignKey, Text


class Eda(BaseDbModel):
    __tablename__ = "ml_eda"
    __table_args__ = ({'comment': 'ML Eda'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    group: Mapped[Optional[str]] = mapped_column(String(64), default='default', comment="Group")
    config: Mapped[Optional[str]] = mapped_column(Text, comment="Config")
    public: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is public")

    # bind to unique dataset
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey('ml_dataset.id'), comment="Dataset id")
    dataset: Mapped[Dataset] = relationship(lazy='selectin')

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')

