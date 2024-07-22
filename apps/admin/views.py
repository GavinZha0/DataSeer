#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : views.py
# @IDE            : PyCharm
# @desc           : Router，View

from core.crud import RET
from utils.response import SuccessResponse
from fastapi import APIRouter, Depends
from apps.auth.auth import Auth, AllUserAuth
from . import model, crud, param


app = APIRouter()


###########################################################
#    SysOrg
###########################################################
@app.post("/org/list", summary="List Sys Orgs")
async def list_org(req: param.OrgParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.OrgDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                            v_where=[model.SysOrg.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/org/get", summary="Get a Sys Org")
async def get_org(org_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.OrgDal(auth.db).get_data(org_id, v_ret=RET.DUMP, v_where=[model.SysOrg.org_id == auth.user.oid])
    return SuccessResponse(data)


###########################################################
#    SysParam
###########################################################
@app.post("/param/list", summary="List Sys Params")
async def list_param(req: param.ParamParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.ParamDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                            v_where=[model.SysParam.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/param/get", summary="Get a Sys Param")
async def get_param(param_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.ParamDal(auth.db).get_data(param_id, v_ret=RET.DUMP, v_where=[model.SysParam.org_id == auth.user.oid])
    return SuccessResponse(data)


###########################################################
#    SysMenu
###########################################################
@app.post("/menu/list", summary="List Sys Menus")
async def list_menu(req: param.MenuParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.MenuDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                            v_where=[model.SysMenu.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/menu/get", summary="Get a Sys Menu")
async def get_menu(menu_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.MenuDal(auth.db).get_data(menu_id, v_ret=RET.DUMP, v_where=[model.SysMenu.org_id == auth.user.oid])
    return SuccessResponse(data)


###########################################################
#    SysRole
###########################################################
@app.post("/role/list", summary="List Sys Roles")
async def list_role(req: param.RoleParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.RoleDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                            v_where=[model.SysRole.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/role/get", summary="Get a Sys Role")
async def get_role(role_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.RoleDal(auth.db).get_data(role_id, v_ret=RET.DUMP, v_where=[model.SysMenu.org_id == auth.user.oid])
    return SuccessResponse(data)


###########################################################
#    SysUser
###########################################################
@app.post("/user/list", summary="List Sys Users")
async def list_user(req: param.UserParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.UserDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                            v_where=[model.SysUser.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/user/get", summary="Get a Sys User")
async def get_user(user_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.UserDal(auth.db).get_data(user_id, v_ret=RET.DUMP, v_where=[model.SysMenu.org_id == auth.user.oid])
    return SuccessResponse(data)

