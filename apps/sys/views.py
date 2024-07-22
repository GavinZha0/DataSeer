# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2021/10/24 16:44
# @File           : views.py
# @IDE            : PyCharm
# @desc           : 主要接口文件

from fastapi import Depends, APIRouter
from apps.auth.auth import Auth, AllUserAuth
from . import crud, schema, param
from utils.response import SuccessResponse
from core.database import db_getter
from sqlalchemy.ext.asyncio import AsyncSession


app = APIRouter()


###########################################################
#    LogAction
###########################################################
@app.post("/logaction/list", summary="List LogAction")
async def list_logaction(p: param.LogactionParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.LogactionDal(auth.db).get_datas(**p.dict(), v_count=True)
    return SuccessResponse(datas, count=count)


@app.post("/logaction/create", summary="Create LogAction")
async def create_logaction(data: schema.Logaction, auth: Auth = Depends(AllUserAuth())):
    return SuccessResponse(await crud.LogactionDal(auth.db).create_data(data=data))



###########################################################
#    LogAccess
###########################################################

@app.post("/logaccess/list", summary="List LogAccess", tags=["LogAccess"])
async def list_logaccess(p: param.LogaccessParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.LogaccessDal(auth.db).get_datas(**p.dict(), v_count=True)
    return SuccessResponse(datas, count=count)


@app.post("/logaccess/create", summary="Create LogAccess", tags=["LogAccess"])
async def create_logaccess(data: schema.Logaccess, auth: Auth = Depends(AllUserAuth())):
    return SuccessResponse(await crud.LogaccessDal(auth.db).create_data(data=data))
