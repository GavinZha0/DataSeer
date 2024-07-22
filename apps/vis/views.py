#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : views.py
# @IDE            : PyCharm
# @desc           : Router，View

from core.crud import RET
from . import model, param, crud
from apps.auth.auth import AllUserAuth
from fastapi import APIRouter, Depends
from utils.response import SuccessResponse
from apps.auth.token import Auth


app = APIRouter()


###########################################################
#    Dataset
###########################################################
@app.post("/dataset/list", summary="List Vis Datasets")
async def list_dataset(req: param.DatasetParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.DatasetDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                        v_where=[model.Dataset.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/dataset/get", summary="Get a Vis Dataset")
async def get_dataset(dataset_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.DatasetDal(auth.db).get_data(dataset_id, v_ret=RET.DUMP, v_where=[model.Dataset.org_id == auth.user.oid])
    return SuccessResponse(data)


###########################################################
#    Dataview
###########################################################
@app.post("/dataview/list", summary="List Vis Views")
async def list_dataview(req: param.DataviewParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.DataviewDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                        v_where=[model.Dataview.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/dataview/get", summary="Get a Vis View")
async def get_dataview(view_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.DataviewDal(auth.db).get_data(view_id, v_ret=RET.DUMP, v_where=[model.Dataview.org_id == auth.user.oid])
    return SuccessResponse(data)


###########################################################
#    Datareport
###########################################################
@app.post("/datareport/list", summary="List Vis Reports")
async def list_datareport(req: param.DatareportParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.DatareportDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                        v_where=[model.Datareport.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/datareport/get", summary="Get a Vis Report")
async def get_datareport(report_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.DatareportDal(auth.db).get_data(report_id, v_ret=RET.DUMP, v_where=[model.Datareport.org_id == auth.user.oid])
    return SuccessResponse(data)

