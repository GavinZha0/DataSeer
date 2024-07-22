#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : views.py
# @IDE            : PyCharm
# @desc           : Router，View

from core.crud import RET
from utils.response import SuccessResponse
from apps.auth.token import Auth
from . import model, param, crud
from apps.auth.auth import AllUserAuth
from fastapi import Depends, APIRouter


app = APIRouter()


###########################################################
#    AiModel
###########################################################
@app.post("/model/list", summary="List AiModel")
async def list_model(req: param.ModelParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.ModelDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                          v_where=[model.AiModel.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/model/get", summary="Get AiModel")
async def get_model(model_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.ModelDal(auth.db).get_data(model_id, v_ret=RET.DUMP, v_where=[model.AiModel.org_id == auth.user.oid])
    return SuccessResponse(data)


###########################################################
#    AiImage
###########################################################
@app.post("/image/list", summary="List AiImage")
async def list_image(req: param.ImageParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.ImageDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                          v_where=[model.AiImage.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/image/get", summary="Get AiImage")
async def get_image(image_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.ImageDal(auth.db).get_data(image_id, v_ret=RET.DUMP, v_where=[model.AiImage.org_id == auth.user.oid])
    return SuccessResponse(data)

