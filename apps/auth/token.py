# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2021/10/24 16:44
# @File           : auth.py
# @IDE            : PyCharm
# @desc           : 用户凭证验证装饰器

import jwt
from pydantic import BaseModel
from config import settings
from sqlalchemy.ext.asyncio import AsyncSession
from core.exception import CustomException
from utils import status
from datetime import datetime, timedelta


class UserInfo(object):
    def __init__(self):
        self.name: str = None
        self.id: int | None = 0
        self.oid: int | None = 0
        self.rid: list | None = []
        self.role: list | None = []


class Auth(BaseModel):
    user: UserInfo = None
    db: AsyncSession

    class Config:
        # 接收任意类型
        arbitrary_types_allowed = True


class AuthToken:
    """
    用于用户每次调用接口时，验证用户提交的token是否正确，并从token中获取用户信息
    """

    # status_code = 401 时，表示强制要求重新登录，因账号已冻结，账号已过期，手机号码错误，刷新token无效等问题导致
    # 只有 code = 401 时，表示 token 过期，要求刷新 token
    # 只有 code = 错误值时，只是报错，不重新登陆
    error_code = status.HTTP_401_UNAUTHORIZED
    warning_code = status.HTTP_ERROR

    # status_code = 403 时，表示强制要求重新登录，因无系统权限，而进入到系统访问等问题导致

    @staticmethod
    def create_token(payload: dict, expires: timedelta = None):
        """
        创建一个生成新的访问令牌的工具函数。
        #TODO 传入的时间为UTC时间datetime.datetime类型，但是在解码时获取到的是本机时间的时间戳
        """
        current_time = datetime.utcnow()
        if expires:
            expire = current_time + expires
        else:
            expire = current_time + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        payload.update({"iat": current_time})
        payload.update({"exp": expire})
        encoded_jwt = jwt.encode(payload, settings.SECRET_KEY, settings.ALGORITHM)
        return encoded_jwt

    @classmethod
    def validate_token(cls, token: str | None) -> str:
        """
        验证用户 token
        """
        if not token:
            raise CustomException(
                msg="请您先登录！",
                code=status.HTTP_403_FORBIDDEN,
                status_code=status.HTTP_403_FORBIDDEN
            )
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user = UserInfo()
            user.name = payload.get("sub")
            user.id = payload.get("jti")
            user.oid = payload.get('iss')
            user.rid = payload.get("rid")
            user.role = payload.get('role')

            if user.id is None or user.name is None or user.oid is None:
                # token doesn't have valid info
                raise CustomException(
                    msg="未认证，请您重新登录",
                    code=status.HTTP_403_FORBIDDEN,
                    status_code=status.HTTP_403_FORBIDDEN
                )
        except (jwt.exceptions.InvalidSignatureError, jwt.exceptions.DecodeError):
            # fail to decode token
            raise CustomException(
                msg="无效认证，请您重新登录",
                code=status.HTTP_403_FORBIDDEN,
                status_code=status.HTTP_403_FORBIDDEN
            )
        except jwt.exceptions.ExpiredSignatureError:
            # token expires
            raise CustomException(msg="认证已过期，请您重新登录", code=cls.error_code, status_code=cls.error_code)
        return user

