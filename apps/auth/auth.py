# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2021/10/24 16:44
# @File           : current.py
# @IDE            : PyCharm
# @desc           : 获取认证后的信息工具

from sqlalchemy.ext.asyncio import AsyncSession
from core.exception import CustomException
from utils import status
from fastapi import Request, Depends
from config import settings
from core.database import db_getter
from apps.auth.token import UserInfo, Auth, AuthToken


class OpenAuth(AuthToken):
    """
    开放认证，无认证也可以访问
    """

    async def __call__(
            self,
            request: Request,
            token: str = Depends(settings.oauth2_scheme),
            db: AsyncSession = Depends(db_getter)
    ):
        if not settings.OAUTH_ENABLE:
            return Auth(db=db)
        try:
            user: UserInfo = self.validate_token(token)
            return Auth(user=user, db=db)
        except CustomException:
            return Auth(db=db)


class AllUserAuth(AuthToken):
    """
    支持所有用户认证
    获取用户基本信息
    """

    async def __call__(
            self,
            request: Request,
            token: str = Depends(settings.oauth2_scheme),
            db: AsyncSession = Depends(db_getter)
    ):
        if not settings.OAUTH_ENABLE:
            return Auth(db=db)
        user: UserInfo = self.validate_token(token)
        return Auth(user=user, db=db)


class FullAdminAuth(AuthToken):
    """
    管理员认证
    获取员工用户完整信息
    """

    async def __call__(
            self,
            request: Request,
            token: str = Depends(settings.oauth2_scheme),
            db: AsyncSession = Depends(db_getter)
    ) -> Auth:
        if not settings.OAUTH_ENABLE:
            return Auth(db=db)
        user: UserInfo = self.validate_token(token)
        if 'admin' not in user.role and 'administrator' not in user.role:
            raise CustomException(msg="无权限操作", code=status.HTTP_403_FORBIDDEN)
        else:
            return Auth(user=user, db=db)
