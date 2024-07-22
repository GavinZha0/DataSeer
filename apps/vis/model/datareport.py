#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @File           : datareport.py
# @IDE            : PyCharm
# @desc           : Datareport

from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from apps.admin.model import SysOrg, SysMenu
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, ForeignKey, Text


class Datareport(BaseDbModel):
    __tablename__ = "viz_report"
    __table_args__ = ({'comment': 'Datareport'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    group: Mapped[Optional[str]] = mapped_column(String(64), default='default', comment="Group")
    type: Mapped[str] = mapped_column(String(64), comment="Type")
    pages: Mapped[str] = mapped_column(Text, comment="Pages")
    public: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is public")
    pub_pub: Mapped[bool] = mapped_column(Boolean, default=True, comment="Is public publish")
    view_ids: Mapped[str] = mapped_column(Text, comment="View ids")

    # bind to unique menu
    menu_id: Mapped[int] = mapped_column(ForeignKey('sys_menu.id'), comment="Menu id")
    menu: Mapped[SysMenu] = relationship()

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')

