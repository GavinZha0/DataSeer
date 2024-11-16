#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/10
# @desc           : AI model
import datetime
from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column, relationship
from apps.admin.model import SysOrg
from db.db_base import BaseDbModel
from sqlalchemy import String, Boolean, Integer, ForeignKey, Text, DateTime, func

class AiModel(BaseDbModel):
    __tablename__ = "ai_model"
    __table_args__ = ({'comment': 'AI model'})

    name: Mapped[str] = mapped_column(String(64), comment="Name")
    desc: Mapped[Optional[str]] = mapped_column(String(128), comment="Description")
    area: Mapped[str] = mapped_column(String(32), default='data', comment="Area")
    tags: Mapped[Optional[str]] = mapped_column(String(128), comment="Tags")
    version: Mapped[int] = mapped_column(Integer, comment="Version")
    algo_id: Mapped[Optional[int]] = mapped_column(Integer, comment="Algo Id")
    rate: Mapped[Optional[int]] = mapped_column(Integer, comment="Rate")
    price: Mapped[Optional[str]] = mapped_column(String(16), comment="Price")
    run_id: Mapped[Optional[str]] = mapped_column(String(32), comment="Run Id")
    deploy_to: Mapped[Optional[str]] = mapped_column(String(64), comment="Deploy To")
    endpoint: Mapped[Optional[str]] = mapped_column(String(64), comment="Endpoint")
    public: Mapped[bool] = mapped_column(Boolean, default=False, comment="Is public")
    status: Mapped[int] = mapped_column(Integer, default=0, comment="0:idle; 1:serving; 2:exception; 3:unknown;")

    deployed_by: Mapped[str] = mapped_column(String(64), nullable=False, comment='Deployed By')
    deployed_at: Mapped[datetime] = mapped_column(DateTime, comment='Deployed At')

    # bind to unique org
    org_id: Mapped[int] = mapped_column(ForeignKey('sys_org.id'), comment="Org id")
    org: Mapped[SysOrg] = relationship(lazy='selectin')