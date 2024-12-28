#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : views.py
# @desc           : ML API
import ray
from sqlalchemy import select
from fastapi import Depends, APIRouter
from apps.auth.auth import AllUserAuth
from core.crud import RET
from utils.data_loader import DataLoader
from . import param, schema, crud, model
from apps.datamgr import crud as dm_crud
from utils.response import SuccessResponse, ErrorResponse
from apps.auth.token import Auth
from itertools import groupby
from .algo.algo_main import train_pipeline, extract_algos, extract_algo_args, extract_algo_scores, \
    extract_frame_versions, extract_ml_datasets
from .dataset.dataset_main import get_data_stat
from .eda.eda_main import eda_build_chart
from .experiment.experiment import exper_reg, exper_unreg, exper_publish, exper_unpublish

app = APIRouter()


###########################################################
#    MlDataset
###########################################################
@app.post("/dataset/list", summary="List Ml Datasets")
async def list_dataset(req: param.DatasetParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.DatasetDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                            v_where=[model.Dataset.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/dataset/groups", summary="Get Ml Dataset groups")
async def get_dataset(auth: Auth = Depends(AllUserAuth())):
    datas = await crud.DatasetDal(auth.db).get_datas(v_start_sql=select(model.Dataset.group), v_distinct=True,
                                                     v_where=[model.Dataset.org_id == auth.user.oid])
    return SuccessResponse({'records': datas})


@app.post("/dataset/tree", summary="Get Ml Dataset tree")
async def tree_dataset(auth: Auth = Depends(AllUserAuth())):
    datas = await crud.DatasetDal(auth.db).get_datas(v_ret=RET.SCHEMA, v_where=[model.Dataset.org_id == auth.user.oid])
    ret_tree = []
    n = 0
    for key, value in groupby(datas, key=lambda x: x.group):
        temp = {'id': 1000+100*n, 'type': 'group', 'name': 'Null' if key is None or len(key) == 0 else key, 'value': 'Null' if key is None or len(key) == 0 else key, 'isLeaf': False, 'selectable': False, 'children': []}
        for k in value:
            temp['children'].append({'id': k.id, 'name': k.name, 'value': k.name, 'isLeaf': True, 'selectable': True})
        ret_tree.append(temp)
        n = n + 1
    return SuccessResponse({'records': ret_tree})


@app.post("/dataset/get", summary="Get a Ml Dataset")
async def get_dataset(req: schema.DatasetGetOne, auth: Auth = Depends(AllUserAuth())):
    data = await crud.DatasetDal(auth.db).get_data(req.id, v_ret=RET.DUMP, v_where=[model.Dataset.org_id == auth.user.oid])
    return SuccessResponse(data)

@app.post("/dataset/extract", summary="Extract Build-in Datasets")
async def extract_buildin_dataset(auth: Auth = Depends(AllUserAuth())):
    data = await extract_ml_datasets()
    return SuccessResponse(data)


@app.post("/dataset/execute", summary="Exe Ml Dataset")
async def exe_dataset(req: schema.DatasetGetOne, auth: Auth = Depends(AllUserAuth())):
    # query dataset and datasource info
    dataset_info = await crud.DatasetDal(auth.db).get_data(req.id, v_ret=RET.SCHEMA, v_where=[model.Dataset.org_id == auth.user.oid])
    src_info = await dm_crud.DatasourceDal(auth.db).get_data(dataset_info.sourceId)

    # initialize data loader and load data
    loader = DataLoader(src_info)
    df = await loader.load(dataset_info.content, dataset_info.variable)
    if df is None:
        return SuccessResponse({'total': 0, 'records': [], 'stat': {}})

    # get statistics info
    total = len(df)
    data, stat = await get_data_stat(dataset_info.type, df, None)
    return SuccessResponse({'total': total, 'records': data, 'stat': stat})


@app.post("/dataset/stat", summary="Get data stat")
async def get_dataset_stat(req: schema.DatasetGetStat, auth: Auth = Depends(AllUserAuth())):
    # query datasource info
    source_info = await dm_crud.DatasourceDal(auth.db).get_data(req.id)

    # initialize data loader and load data
    loader = DataLoader(source_info)
    df = await loader.load(req.content, req.variable)
    if df is None:
        return SuccessResponse({'total': 0, 'records': [], 'stat': {}})

    # get statistics info
    total = len(df)
    data, stat = await get_data_stat(req.type, df, req.limit)
    return SuccessResponse({'total': total, 'records': data, 'stat': stat})


###########################################################
#    MlEda
###########################################################
@app.post("/eda/list", summary="List Ml Edas")
async def list_eda(req: param.EdaParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.EdaDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                        v_where=[model.Eda.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/eda/get", summary="Get a Ml Eda")
async def get_algo(eda_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.EdaDal(auth.db).get_data(eda_id, v_ret=RET.DUMP, v_where=[model.Eda.org_id == auth.user.oid])
    return SuccessResponse(data)


@app.post("/eda/groups", summary="Get Ml Eda groups")
async def get_dataset(auth: Auth = Depends(AllUserAuth())):
    datas = await crud.EdaDal(auth.db).get_datas(v_start_sql=select(model.MlEda.group), v_distinct=True,
                                                 v_where=[model.Eda.org_id == auth.user.oid])
    return SuccessResponse({'records': datas})


@app.post("/eda/build", summary="Build ML Eda charts")
async def build_eda(req: schema.EdaBuildParam, auth: Auth = Depends(AllUserAuth())):
    # query dataset and datasource info
    dataset_info = await crud.DatasetDal(auth.db).get_data(req.dataset_id, v_ret=RET.SCHEMA,
                                                           v_where=[model.Dataset.org_id == auth.user.oid])
    source_info = await dm_crud.DatasourceDal(auth.db).get_data(dataset_info.sourceId)

    # initialize data loader and load data
    loader = DataLoader(source_info, dataset_info)
    # load and transform data
    df = await loader.run()
    if df is None:
        return SuccessResponse()

    # generate eda chart
    json_rsp = await eda_build_chart(req.tier, req.kind, req.config, df, dataset_info.fields)
    return SuccessResponse(json_rsp)


@app.post("/eda/execute", summary="Execute ML Eda analysis")
async def execute_eda(eda_id: int, auth: Auth = Depends(AllUserAuth())):
    return SuccessResponse(await crud.EdaDal(auth.db).get_data(eda_id))


###########################################################
#    MlAlgo
###########################################################
@app.post("/algo/list", summary="List Ml Algos")
async def list_algo(req: param.AlgoParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.AlgoDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                         v_where=[model.Algo.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/algo/get", summary="Get a Ml Algo")
async def get_algo(algo_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.AlgoDal(auth.db).get_data(algo_id, v_ret=RET.DUMP, v_where=[model.Algo.org_id == auth.user.oid])
    return SuccessResponse(data)


@app.post("/algo/vers", summary="Get framework versions")
async def get_versions(auth: Auth = Depends(AllUserAuth())):
    ver_json = await extract_frame_versions()
    return SuccessResponse(ver_json)


@app.post("/algo/algos", summary="Get Existing algos")
async def get_algos(params: schema.AlgoGetParam, auth: Auth = Depends(AllUserAuth())):
    algo_json = await extract_algos(params.category)
    return SuccessResponse(algo_json)

@app.post("/algo/args", summary="Get algo args")
async def get_args(params: schema.AlgoGetArgsParam, auth: Auth = Depends(AllUserAuth())):
    args_and_doc = await extract_algo_args(params.category, params.algo)
    return SuccessResponse({'records': args_and_doc})


@app.post("/algo/scores", summary="Get algo scores")
async def get_args(params: schema.AlgoGetParam, auth: Auth = Depends(AllUserAuth())):
    algo_scores = await extract_algo_scores(params.category)
    return SuccessResponse(algo_scores)


@app.post("/algo/execute", summary="train algo")
async def execute_algo(req: schema.AlgoExeParam, auth: Auth = Depends(AllUserAuth())):
    result = await train_pipeline(req.id, auth.db, auth.user)
    if result is False:
        return ErrorResponse('Failed to execute algo training')
    else:
        return SuccessResponse()


###########################################################
#    MlFlow
###########################################################
@app.post("/flow/list", summary="List Ml Flows")
async def list_flow(req: param.FlowParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.FlowDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                         v_where=[model.Flow.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/flow/get", summary="Get a Ml Flow")
async def get_flow(data_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.FlowDal(auth.db).get_data(data_id, v_ret=RET.DUMP, v_where=[model.Flow.org_id == auth.user.oid])
    return SuccessResponse(data)


@app.post("/experiment/reg", summary="Register a model")
async def reg_experiment(req: schema.AlgoExperReg, auth: Auth = Depends(AllUserAuth())):
    reg_ver = await exper_reg(req.trialId, req.algoName, req.algoId, auth.user.id)
    if reg_ver>0:
        return SuccessResponse(dict(version=reg_ver))
    else:
        return ErrorResponse()

@app.post("/experiment/unreg", summary="Un-register a model")
async def reg_experiment(req: schema.AlgoExperUnreg, auth: Auth = Depends(AllUserAuth())):
    await exper_unreg(req.algoId, req.version, auth.user.id)
    return SuccessResponse()


@app.post("/experiment/publish", summary="Publish a model")
async def publish_experiment(req: schema.AlgoExperReg, auth: Auth = Depends(AllUserAuth())):
    reg_ver, published = await exper_publish(req.trialId, req.algoName, req.algoId, auth.user.id)
    if reg_ver>0 and published:
        return SuccessResponse(dict(version=reg_ver, published=published))
    else:
        return ErrorResponse()

@app.post("/experiment/unpublish", summary="Un-publish a model")
async def unpublish_experiment(req: schema.AlgoExperUnreg, auth: Auth = Depends(AllUserAuth())):
    await exper_unpublish(req.algoId, req.version, auth.user.id)
    return SuccessResponse()



