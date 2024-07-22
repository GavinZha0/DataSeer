#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : dataview.py
# @IDE            : PyCharm
# @desc           : Dataview

from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from apps.admin.model import SysOrg
from apps.vis.model import Dataset
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, Integer, ForeignKey, Text


class Dataview(BaseDbModel):
    __tablename__ = "viz_view"
    __table_args__ = ({'comment': 'Dataview'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    group: Mapped[Optional[str]] = mapped_column(String(64), default='default', comment="Group")
    type: Mapped[str] = mapped_column(String(64), comment="Type")
    dim: Mapped[str] = mapped_column(String(255), comment="Dimension")
    relation: Mapped[Optional[str]] = mapped_column(String(255), comment="Relation")
    location: Mapped[Optional[str]] = mapped_column(String(255), comment="Location")
    metrics: Mapped[str] = mapped_column(String(255), comment="Metrics")
    agg: Mapped[Optional[str]] = mapped_column(String(16), comment="Aggression")
    prec: Mapped[Optional[int]] = mapped_column(Integer, comment="Precision")
    filter: Mapped[Optional[str]] = mapped_column(Text, comment="Filter")
    sorter: Mapped[Optional[str]] = mapped_column(Text, comment="Sorter")
    variable: Mapped[Optional[str]] = mapped_column(Text, comment="Variable")
    calculation: Mapped[Optional[str]] = mapped_column(Text, comment="Calculation")
    model: Mapped[Optional[str]] = mapped_column(Text, comment="Model")
    lib_name: Mapped[str] = mapped_column(String(16), comment="Lib name")
    lib_ver: Mapped[Optional[str]] = mapped_column(String(16), comment="Lib version")
    lib_cfg: Mapped[Optional[str]] = mapped_column(Text, comment="Lib config")
    public: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is public")

    # bind to a unique dataset
    dataset_id: Mapped[int] = mapped_column(ForeignKey('viz_dataset.id'), comment="Dataset id")
    dataset: Mapped[Dataset] = relationship(back_populates="dataviews")

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')

