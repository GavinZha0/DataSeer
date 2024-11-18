#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @desc           : Router，View

from core.crud import RET
from utils.response import SuccessResponse, ErrorResponse
from apps.auth.token import Auth
from . import model, param, crud
from apps.auth.auth import AllUserAuth
from fastapi import Depends, APIRouter

from .mstore.data import data_execute
from .schema.data import AiDataExecute
from .schema.model import AiModelDeploy
from .mstore.model import model_deploy, model_terminate

app = APIRouter()


###########################################################
#    AiModel
###########################################################
@app.post("/model/list", summary="List Ai Models")
async def list_model(req: param.ModelParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.ModelDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                          v_where=[model.AiModel.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)

AI_MODEL_STATUS_IDLE = 0
AI_MODEL_STATUS_SERVING = 1
AI_MODEL_STATUS_EXCEPTION = 2
AI_MODEL_STATUS_UNKNOWN = 3

@app.post("/model/deploy", summary="Deploy Ai Model")
async def deploy_model(req: AiModelDeploy, auth: Auth = Depends(AllUserAuth())):
    data = await crud.ModelDal(auth.db).get_data(req.id, v_ret=RET.SCHEMA, v_where=[model.AiModel.org_id == auth.user.oid])
    if data.deployTo != 'Docker' and data.status > AI_MODEL_STATUS_IDLE:
        # terminate service
        print(f'terminate ML service {req.id} successfully!')
        await model_terminate(data.deployTo, data.endpoint)
        await crud.ModelDal(auth.db).put_data(req.id, dict(status=AI_MODEL_STATUS_IDLE))
        # stop service successfully
        return SuccessResponse(dict(status=AI_MODEL_STATUS_IDLE))

    # docker image name must be lowercase
    img_name = data.name.replace(' ', '_')
    img_name = f'{img_name}_{auth.user.id}'
    img_name = img_name.lower()
    # deploy model and start serving
    rst, info = await model_deploy(data.runId, data.deployTo, data.endpoint, img_name)
    if rst == AI_MODEL_STATUS_UNKNOWN:
        print(f'ML service {req.id} deployed but status is unknown!')
        await crud.ModelDal(auth.db).put_data(req.id, dict(status=AI_MODEL_STATUS_UNKNOWN, endpoint=info))
        return SuccessResponse(dict(status=AI_MODEL_STATUS_UNKNOWN, endpoint=info))
    elif rst == AI_MODEL_STATUS_SERVING:
        # start service successfully
        print(f'ML service {req.id} is serving!')
        await crud.ModelDal(auth.db).put_data(req.id, dict(status=AI_MODEL_STATUS_SERVING, endpoint=info))
        return SuccessResponse(dict(status=AI_MODEL_STATUS_SERVING, endpoint=info))
    else:
        # fail to start service
        print(f'Failed to start ML service {req.id}!')
        return ErrorResponse(info)

###########################################################
#    AiData
###########################################################
@app.post("/data/list", summary="List AiData")
async def list_data(req: param.ImageParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.ImageDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                          v_where=[model.AiImage.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/data/execute", summary="Execute AiData")
async def execute_data(req: AiDataExecute, auth: Auth = Depends(AllUserAuth())):
    result = await data_execute(req.endpoint, req.data)
    return SuccessResponse(result)


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

