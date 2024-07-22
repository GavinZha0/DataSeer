#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/02/12
# @File           : views.py
# @IDE            : PyCharm
# @desc           : Router，View
import base64
from itertools import groupby
from operator import itemgetter
from fastapi import APIRouter, Depends

from core.crud import RET
from utils.db_executor import DbExecutor
from . import param, schema, crud, model
from apps.auth.auth import Auth, AllUserAuth
from utils.response import SuccessResponse
from sqlalchemy import inspect, select

app = APIRouter()


###########################################################
#    Datasource
###########################################################
@app.post("/datasource/list", summary="List Datasource")
async def list_datasource(req: param.DatasourceParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    datas, count = await crud.DatasourceDal(auth.db).get_datas(**req.dict(), v_count=True, v_ret=RET.DUMP,
                                                               v_where=[model.Datasource.org_id == auth.user.oid])
    return SuccessResponse(datas, count=count)


@app.post("/datasource/groups", summary="Get Datasource groups")
async def get_dataset(auth: Auth = Depends(AllUserAuth())):
    datas = await crud.DatasourceDal(auth.db).get_datas(v_start_sql=select(model.Datasource.group), v_distinct=True,
                                                        v_where=[model.Datasource.org_id == auth.user.oid])
    return SuccessResponse({'records': datas})


@app.post("/datasource/tree", summary="Get Datasource tree")
async def tree_datasource(auth: Auth = Depends(AllUserAuth())):
    datas = await crud.DatasourceDal(auth.db).get_datas(v_where=[model.Datasource.org_id == auth.user.oid])
    ret_tree = []
    n = 0
    for key, value in groupby(datas, key=itemgetter('group')):
        temp = {'id': 1000+100*n, 'type': 'group', 'name': 'Null' if key is None or len(key) == 0 else key, 'value': 'Null' if key is None or len(key) == 0 else key, 'isLeaf': False, 'selectable': False, 'children': []}
        for k in value:
            temp['children'].append({'id': k['id'], 'name': k['name'], 'value': k['name'], 'isLeaf': True, 'selectable': True})
        ret_tree.append(temp)
        n = n + 1
    return SuccessResponse({'records': ret_tree})


@app.delete("/datasource/delete", summary="Delete Datasource")
async def delete_datasource(source_id: int, auth: Auth = Depends(AllUserAuth())):
    await crud.DatasourceDal(auth.db).delete_data(source_id, v_where=[model.Datasource.org_id == auth.user.oid])
    return SuccessResponse("delete successfully")


@app.post("/datasource/get", summary="Get a Datasource")
async def get_datasource(source_id: int, auth: Auth = Depends(AllUserAuth())):
    data = await crud.DatasourceDal(auth.db).get_data(source_id, v_ret=RET.DUMP, v_where=[model.Datasource.org_id == auth.user.oid])
    return SuccessResponse(data)


@app.post("/datasource/tables", summary="Get Datasource tree")
async def get_datasource_tables(p: param.SrcTableParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    src_info = await crud.DatasourceDal(auth.db).get_data(p.id, v_schema=schema.DatasourceSimpleOut, v_where=[
        model.Datasource.org_id == auth.user.oid])
    passport = src_info['username'] + ':' + base64.b64decode(src_info['password']).decode('utf-8')
    db = DbExecutor(src_info['type'], src_info['url'], passport, src_info['params'])
    tables = inspect(db.engine).get_table_names()
    return SuccessResponse({'records': [{'id': idx, 'name': name} for idx, name in enumerate(tables)]})


@app.post("/datasource/execute", summary="Exe sql on a source ")
async def datasource_exe_sql(p: param.ExeSqlParams = Depends(), auth: Auth = Depends(AllUserAuth())):
    # Source should be in same org with you
    src_info = await crud.DatasourceDal(auth.db).get_data(p.id, v_schema=schema.DatasourceSimpleOut,
                                                          v_where=[model.Datasource.org_id == auth.user.oid])
    passport = src_info['username'] + ':' + base64.b64decode(src_info['password']).decode('utf-8')
    db = DbExecutor(src_info['type'], src_info['url'], passport, src_info['params'])
    df, total = db.db_query(p.sql, None, p.limit)
    cols = [{'id': idx, 'name': col, 'type': df[col].dtype.name} for idx, col in enumerate(df)]
    ts_col = df.select_dtypes(include='datetime').columns.tolist()
    if ts_col is not None:
        for col in ts_col:
            df[col] = df[col].dt.strftime('%m/%d/%Y %H:%M:%S')
    return SuccessResponse({'total': total, 'metadata': cols, 'records': df.values.tolist()})
