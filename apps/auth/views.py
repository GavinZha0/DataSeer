# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2021/10/24 16:44
# @File           : views.py
# @IDE            : PyCharm
# @desc           : 安全认证视图

"""
JWT 表示 「JSON Web Tokens」。https://jwt.io/
它是一个将 JSON 对象编码为密集且没有空格的长字符串的标准。
通过这种方式，你可以创建一个有效期为 1 周的令牌。然后当用户第二天使用令牌重新访问时，你知道该用户仍然处于登入状态。
一周后令牌将会过期，用户将不会通过认证，必须再次登录才能获得一个新令牌。
"""

import re
import typing
from datetime import timedelta
from fastapi.responses import ORJSONResponse
from fastapi import APIRouter, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from core.crud import RET
from core.database import db_getter
from core.exception import CustomException
from utils.response import SuccessResponse
from config import settings
from apps.admin.crud import UserDal as SysUserDal
from apps.admin.crud import MenuDal as SysMenuDal
from .auth import AllUserAuth
from apps.auth.token import Auth, AuthToken
from apps.sys.model.access import LogAccess

app = APIRouter()
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


@app.post("/login", summary="Login", description="User login")
async def login(
        request: Request,
        data: OAuth2PasswordRequestForm = Depends(),
        db: AsyncSession = Depends(db_getter)
):
    # it can be phone number, email or username
    uid = data.username
    if uid.isdigit():
        # phone number
        user = await SysUserDal(db).get_data(phone=uid)
    elif re.match("\w+@\w+\.\w+", uid, re.ASCII):
        # email
        user = await SysUserDal(db).get_data(email=uid)
    else:
        # username
        user = await SysUserDal(db).get_data(name=uid)

    if not user:
        raise CustomException(status_code=401, code=401, msg="User doesn't exist")

    result = pwd_context.verify(data.password, user.password)
    if not result:
        raise CustomException(status_code=401, code=401, msg="Failed to authorize")
    if not user.active:
        raise CustomException(status_code=401, code=401, msg="User is inactive")

    user_in_token = {'jti': str(user.id), 'sub': user.name, 'iss': str(user.org.id), 'rid': [], 'role': []}
    user_in_token['rid'] = [role.id for role in user.roles]
    user_in_token['role'] = [role.name for role in user.roles]

    access_token = AuthToken.create_token(user_in_token)
    shadow_token = AuthToken.create_token(user_in_token, timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 2))

    # respond user info
    resp = {
          "id": user.id,
          "name": user.name,
          "realname": user.realname,
          "avatar": user.avatar,
          "orgId": user.org.id,
          "orgName": "Free Center",
          "roleId": user_in_token,
          "roleName": user_in_token['role'],
          "access_token": access_token
        }

    # put token in header
    headers: typing.Mapping[str, str] = {
        settings.ACCESS_TOKEN_FIELD: access_token,
        settings.SHADOW_TOKEN_FIELD: shadow_token
    }

    await LogAccess.create_login_record(db, user.id, user.name, True, request, resp)
    return ORJSONResponse(content=resp, headers=headers)


@app.post("/permit", summary="Get permit", description="Get permitted menus")
async def get_permitted_menus(auth: Auth = Depends(AllUserAuth())):
    menu_tree = await SysMenuDal(auth.db).get_routers()
    return SuccessResponse({'records': menu_tree})


@app.post("/info", summary="Get info", description="Get logined user info")
async def get_user_info(auth: Auth = Depends(AllUserAuth())):
    data = await SysUserDal(auth.db).get_data(auth.user.id, v_ret=RET.DUMP)
    return SuccessResponse(data)
