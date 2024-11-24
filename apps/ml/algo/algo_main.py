import sys
import base64
import operator
import re
from datetime import datetime
import importlib
import execjs
import pandas as pd
import ray
import mlflow
import torch
import torchvision
from pandas import CategoricalDtype
from sqlalchemy.ext.asyncio import AsyncSession
from apps.ml.algo.algo_trainable import build_ray_trainable
from config import settings
from core.crud import RET
from utils.db_executor import DbExecutor
from apps.datamgr import crud as dm_crud
from apps.ml import crud as ml_crud, schema as ml_schema
import sklearn
from sklearn import preprocessing as pp
from sklearn import feature_extraction as fe
from sklearn import metrics
from sklearn.utils import all_estimators
import sklearn.datasets as skds
import xgboost as xgb
import lightgbm
import lightning
from utils.ray.ray_dataloader import RayDataLoader
from utils.ray.ray_trainer import RayTrainer
import inspect
import ast


"""
return pandas dataframe
"""


async def extract_data(dataset_id: int, db: AsyncSession):
    # get dataset and datasource info
    dataset_info = await ml_crud.DatasetDal(db).get_data(dataset_id, v_ret=RET.SCHEMA)
    source_info = await dm_crud.DatasourceDal(db).get_data(dataset_info.sourceId, v_ret=RET.SCHEMA)

    # connect to target db
    passport = source_info.username + ':' + base64.b64decode(source_info.password).decode('utf-8')
    db = DbExecutor(source_info.type, source_info.url, passport, source_info.params)

    # query data and get dataframe of Pandas
    dataframe, total = await db.db_query(dataset_info.content, None, dataset_info.variable)
    return dataframe, dataset_info.fields, dataset_info.transform


"""
transform data
"""


async def transform_data(df: pd.DataFrame, fields, transform):
    for it in fields:
        if it.get('omit'):
            # delete omit fields
            df.drop(columns=[it['name']], inplace=True)
            continue
        if it.get('attr') == 'date':
            # format datetime
            df[it['name']] = pd.to_datetime(df[it['name']])
            continue
        if it.get('attr') == 'cat':
            # convert type to category
            if it.get('values'):
                cat_type = CategoricalDtype(categories=it.get('values'))
            else:
                u_values = df[it['name']].value_counts().index.to_list()
                it['values'] = u_values
                cat_type = CategoricalDtype(categories=u_values)
            df[it['name']] = df[it['name']].astype(cat_type)
            continue
        if (it.get('type') == 'string' or df[it['name']].dtype == 'object') and it.get('attr') == 'conti':
            # convert string to integer
            df[it['name']] = pd.to_numeric(df[it['name']], errors='coerce')
            continue
        if (it.get('type') == 'string' or df[it['name']].dtype == 'object') and it.get('attr') == 'disc':
            # convert string to integer
            df[it['name']] = pd.to_numeric(df[it['name']], errors='coerce', downcast="integer")
            continue

    # process missing value
    missing_values = ["n/a", "na", "--"]
    miss_fields = [it for it in fields if it.get('miss')]
    for it in miss_fields:
        field_name = it['name']
        if df[field_name].isnull().any():
            match it['miss']:
                case 'drop':
                    # drop the row when this field has na
                    df.dropna(subset=[it['name']], inplace=True)
                case 'mean':
                    df[field_name] = df[field_name].fillna(df[field_name].mean())
                case 'median':
                    df[field_name] = df[field_name].fillna(df[field_name].median())
                case 'mode':
                    df[field_name] = df[field_name].fillna(df[field_name].mode())
                case 'min':
                    df[field_name] = df[field_name].fillna(df[field_name].min())
                case 'max':
                    df[field_name] = df[field_name].fillna(df[field_name].max())
                case 'prev':
                    df[field_name] = df[field_name].fillna(method='ffill')
                case 'next':
                    df[field_name] = df[field_name].fillna(method='bfill')
                case 'zero':
                    df[field_name] = df[field_name].fillna(value=0)
                case '_':
                    # assigned value
                    df[field_name] = df[field_name].fillna(it['miss'])

    # drop the row/column if all values are na
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # drop all duplicate rows
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # encoding
    encode_fields = [it for it in fields if it.get('encode')]
    for it in encode_fields:
        field_name = it['name']
        match it['encode']:
            case 'ordinal':  # Ordinal
                if it.get('target'):
                    # encode it to index based on unique values
                    cat_type = CategoricalDtype(categories=it.get('values'))
                    df[field_name] = df[field_name].astype(cat_type).cat.codes
                    # convert it to category again from int8
                    df[field_name] = df[field_name].astype('category')
                else:
                    df[field_name] = pp.OrdinalEncoder().fit_transform(df[field_name])
            case 'hot':  # One-Hot
                df[field_name] = pp.OneHotEncoder().fit_transform(df[field_name])
            case 'hash':  # Hashing
                df[field_name] = fe.FeatureHasher(input_type='string').fit_transform(df[field_name])
            case 'binary':  # Binary
                df[field_name] = pp.Binarizer(threshold=1).fit_transform(df[field_name])
            case 'bins':  # Binning
                df[field_name] = pp.KBinsDiscretizer(n_bins=10, strategy='uniform', encode='ordinal').fit_transform(
                    df[field_name])
            case 'count':  # Count Encode
                df[field_name] = pp.LabelEncoder().fit_transform(df[field_name])
            case 'mean':  # Mean Encode
                df[field_name] = pp.LabelEncoder().fit_transform(df[field_name])
            case 'woe':  # woe Encode
                df[field_name] = pp.LabelEncoder().fit_transform(df[field_name])

    # scaling
    scale_fields = [it for it in fields if it.get('scale')]
    for it in scale_fields:
        field_name = it['name']
        match it['scale']:
            case 'std':
                # mean = 0, stddev = 1
                df[[field_name]] = pp.StandardScaler().fit_transform(df[[field_name]])
            case 'minmax':
                # [0, 1]
                df[field_name] = pp.MinMaxScaler().fit_transform(df[field_name])
            case 'maxabs':
                # [-1, 1]
                df[field_name] = pp.MaxAbsScaler().fit_transform(df[field_name])
            case 'robust':
                df[field_name] = pp.RobustScaler().fit_transform(df[field_name])
            case 'l1':
                df[field_name] = pp.Normalizer(norm='l1').fit_transform(df[field_name])
            case 'l2':
                df[field_name] = pp.Normalizer(norm='l2').fit_transform(df[field_name])
            case '_':
                scale = None
                # do nothing

    return df


"""
extract framework versions
"""
async def extract_frame_versions():
    versions = dict(sklearn=sklearn.__version__, xgboost=xgb.__version__,
                   lightgbm=lightgbm.__version__, lightning=lightning.__version__,
                   ray=ray.__version__, mlflow=mlflow.__version__)
    versions['python'] = sys.version.split('|')[0].strip()
    versions['pytorch'] = torch.__version__.split('+')[0].strip()
    return versions


"""
extract existing dataset
sklearn and pytorch
"""
async def extract_ml_datasets():
    all_set = []
    sklearn_set = dict(id=-1, name='sklearn', children=[])
    sklearn_set['children'] = [{"id": -100-k, "name": v} for k, v in enumerate(skds.__all__) if v.startswith('load') or v.startswith('fetch')]
    all_set.append(sklearn_set)

    torch_set = dict(id=-2, name='pytorch', children=[])
    set_list = list(torchvision.datasets.__all__)
    torch_set['children'] = [{"id": -200-k, "name": v} for k, v in enumerate(set_list)]
    all_set.append(torch_set)
    # imagenet_data = torchvision.datasets.KMNIST('./', train=True, download=True)
    # yesno_data = torchaudio.datasets.YESNO('./', download=True)
    return all_set


"""
extract existing algorithms
framework: sklearn, pytorch, tensorflow
sklearn category: 'classifier', 'regressor', 'transformer', 'cluster'
pytorch category: vision, 
"""
async def extract_algos(framework: str, category: str):
    category_map = {'clf': 'classifier', 'reg': 'regressor', 'cluster': 'cluster', 'transform': 'transformer'}
    result = {}
    match framework:
        case 'sklearn':
            estimators = all_estimators(type_filter=category_map[category])
            for name, class_ in estimators:
                m_name = class_.__dict__.get('__module__').split(".")[1]
                cls_name = class_.__name__
                if m_name != 'dummy':
                    if result.get(m_name) is None:
                        result[m_name] = [cls_name]
                    else:
                        result[m_name].append(cls_name)

        case 'pytorch':
            models = torch.hub.list(f'pytorch/{category}')
            for md in models:
                if md.startswith('get_') or md.find('_') < 0:
                    # ex: 'get_weight', 'vgg11'
                    continue
                # handle modules with '_' as first priority
                # # ex: deeplabv3_mobilenet_v3_large
                md_segs = md.split("_")
                m_name = md_segs[0]
                if result.get(m_name) is None:
                    # add new module to result
                    result[m_name] = [md]
                else:
                    # add sub cat to module
                    result[m_name].append(md)

            for md in models:
                if md.startswith('get_') or md.find('_') >= 0:
                    # ex: 'get_weight', 'convnext_small', 'vgg11_bn'
                    continue
                # handle modules without '_' as second priority
                if result.get(md) is not None:
                    # ex: 'vgg11'
                    result[md].append(md + '_basic')
                else:
                    # end with digits
                    match = re.search(r"\d+$", md)
                    if match:
                        # ex: 'resnet101
                        m_cat = match.group()
                        m_name = md[:0 - len(m_cat)]
                        if result.get(m_name) is None:
                            result[m_name] = [md]
                        else:
                            result[m_name].append(md)
                    else:
                        # ex: alexnet
                        result[md] = []

            # reduce levels if a module has only one sub cat
            del_list = []
            add_list = []
            for m_name in result:
                if len(result[m_name]) == 0:
                    del_list.append(m_name)
                    add_list.append(m_name)
                elif len(result[m_name]) == 1:
                    # only one sub item
                    new_name = result[m_name][0]
                    if result.get(new_name):
                        # add to existing module as basic cat if new_name exists
                        result[new_name].append(new_name + '_basic')
                    else:
                        del_list.append(m_name)
                        add_list.append(new_name)

            # delete old modules
            [result.pop(k) for k in del_list]
            # add new to ungrouped
            result['ungrouped'] = []
            [result['ungrouped'].append(k) for k in add_list]

    # sort result
    result_json = dict(sorted(result.items(), key=operator.itemgetter(0)))
    return result_json


"""
extract arguments of algorithm
framework: sklearn, pytorch, tensorflow
sklearn category: 'classifier', 'regressor', 'transformer', 'cluster'
pytorch category: vision, 
"""
async def extract_algo_args(framework: str, category: str, algo: str):
    category_map = {'clf': 'classifier', 'reg': 'regressor', 'cluster': 'cluster', 'transform': 'transformer'}
    args = []
    algo_func = None
    algo_doc = None
    match framework:
        case 'sklearn':
            # find algo func from sklearn estimators
            # can be ignored arguments
            ignored = ['estimator', 'verbose', 'n_jobs', 'random_state', 'max_iter']
            estimators = all_estimators(type_filter=category_map[category])
            algo_funcs = [class_ for name, class_ in estimators if name == algo]
            algo_func = algo_funcs[0]
            if algo_func:
                # cut doc to get parameter description
                algo_doc = algo_func.__doc__
                print(algo_doc)
                para_idx = algo_doc.find('------\n')
                if para_idx >= 0:
                    # ex: 'Parameters\n ----------\n'
                    algo_doc = algo_doc[para_idx + 8:]

                para_idx = algo_doc.find('------\n')
                if para_idx > 0:
                    # ex: 'Attributes\n --------\n'
                    algo_doc = algo_doc[:para_idx - 20]

        case 'pytorch':
            # can be ignored arguments
            ignored = ['kwargs']
            algo_func = torchvision.models.get_model_builder(algo)
            if algo_func:
                # cut doc to get parameter description
                algo_doc = algo_func.__doc__
                print(algo_doc)
                para_idx = algo_doc.find('Args:\n')
                if para_idx >= 0:
                    # ex: 'Args:\n'
                    algo_doc = algo_doc[para_idx + 6:]

                para_idx = algo_doc.find('.. ')
                if para_idx > 0:
                    # ex: '.. autoclass::'
                    algo_doc = algo_doc[:para_idx - 4]
        case 'default':
            return []

    if algo_func:
        # get arg names and default values of the algo
        params = inspect.signature(algo_func)
        args = [dict(name=it.name, default=it.default) for it in list(params.parameters.values())
                    if (not inspect.isclass(it.default)) and (it.name not in ignored)]

        # get options from doc of algo function
        for it in args:
            # find description of the algo parameter
            # ex: criterion : {"gini", "entropy", "log_loss"}, default="gini"
            idx = algo_doc.find(it["name"] + ' : {')
            idz = algo_doc.find('}', idx)
            if idx > 0 and idz > idx:
                # extract options of the algo
                idy = algo_doc.find('{', idx)
                # remove unnecessary chars and convert to list
                tmp = algo_doc[idy + 1:idz].replace('"', '').replace(' ', '')
                it['options'] = tmp.split(',')

    return dict(algo=algo, args=args, doc=algo_doc)


"""
extract arguments of algorithm
framework: sklearn, pytorch, tensorflow
sklearn category: 'classifier', 'regressor', 'transformer', 'cluster'
pytorch category: vision, 
"""
async def extract_algo_scores(framework: str, category: str):
    category_map = {'clf': 'classifier', 'reg': 'regressor', 'cluster': 'cluster', 'transform': 'transformer'}
    # silhouette_score and calinski_harabasz_score can't get from get_scorer_names()
    # they don't need target value to evaluate cluster. all others need.
    scores = dict(clf=[], reg=[], cluster=[])
    match framework:
        case 'sklearn':
            names = metrics.get_scorer_names()
            scores['cluster'] = [it for it in names if
                              'rand_' in it or 'info_' in it or '_measure_' in it or 'homogeneity' in it
                                or 'fowlkes' in it or 'completeness' in it]
            scores['cluster'].append('silhouette_score')
            scores['cluster'].append('calinski_harabasz_score')
            scores['reg'] = [it for it in names if
                          '_mean_' in it or '_variance' in it or '_error' in it or 'r2' in it or 'd2_' in it]
            scores['clf'] = [it for it in names if it not in scores['cluster'] and it not in scores['reg']]
        case 'pytorch':
            scores['vision'] = []
            scores['audio'] = []

    return scores[category]

"""
build train pipeline
"""


async def train_pipeline(algo_id: int, db: AsyncSession, user: dict):
    # get algo, dataset and datasource info from db
    algo_info = await ml_crud.AlgoDal(db).get_data(algo_id, v_ret=RET.SCHEMA)
    dataset_info = await ml_crud.DatasetDal(db).get_data(algo_info.dataCfg.get('datasetId'), v_ret=RET.SCHEMA)
    source_info = await dm_crud.DatasourceDal(db).get_data(dataset_info.sourceId, v_ret=RET.SCHEMA)
    targets = [field['name'] for field in dataset_info.fields if 'target' in field and 'omit' not in field]

    # build parameters
    params = await build_params(algo_info, user)

    ready = await build_ray_trainable(algo_info.framework, algo_info.srcCode, params)
    if ready is False:
        return False

    train_func = None
    # import module from saved file
    dy_module = importlib.import_module(f'temp.ml.{params["module_name"]}')
    # train_cls = getattr(dy_module, 'RayTrainable')
    # train_func = train_cls.train


    # detect classes
    dy_cls = [cls for name, cls in inspect.getmembers(dy_module, inspect.isclass) if params["module_name"] in cls.__module__]
    if dy_cls is None or len(dy_cls) == 0:
        return False
    elif dy_cls and len(dy_cls) > 0:
        for cls in dy_cls:
            dy_func = [func for name, func in inspect.getmembers(cls, inspect.isfunction) if name in ['train'] and cls.__name__ in func.__qualname__]
            if dy_func and len(dy_func) > 0:
                train_func = dy_func[0]
                break

    # return if no trainable function for Ray
    if train_func is None:
        return False

    # load and transform data
    loader = RayDataLoader.remote(source_info)
    dataset_df = loader.load.remote(dataset_info.content, dataset_info.variable)
    train_eval_data = loader.transform.remote(algo_info.framework, dataset_df, targets, dataset_info.fields,
                                              algo_info.dataCfg['evalRatio'], algo_info.dataCfg['shuffle'])

    # train model
    trainer = RayTrainer.remote(algo_info.framework)
    trainer.train.remote(params, train_func, train_eval_data)

"""
build parameters for train
"""


async def build_params(algo: ml_schema.Algo, user: dict):
    now_ts = int(datetime.now().timestamp())
    # basic parameters
    # use mysql db as tracking store
    # use AWS S3 as file store
    # an experiment binds to an algo, experiment name is unique
    # every run has user info
    params = dict(algo_id=algo.id,
                  algo_name=algo.name,
                  user_id=user.id,
                  user_name=user.name,
                  org_id=user.oid,
                  module_name=f"algo_{algo.id}",
                  tracking_url=settings.SQLALCHEMY_MLFLOW_DB_URL,
                  s3_url=settings.AWS_S3_ENDPOINT,
                  s3_id=settings.AWS_S3_ACCESS_ID,
                  s3_key=settings.AWS_S3_SECRET_KEY,
                  artifact_location=f"s3://datapie-{algo.orgId}/ml/algo_{algo.id}",
                  exper_name=f"algo_{algo.id}_{now_ts}",
                  tune_param=dict(user_id=user.id, epochs=1, tracking_url=settings.SQLALCHEMY_MLFLOW_DB_URL)
                  )

    # get model class name from srcCode
    params['cls_name'] = [f.name for f in ast.parse(algo.srcCode).body if isinstance(f, ast.ClassDef)]

    tuner = params['tune_param']
    if algo.trainCfg:
        params['gpu'] = algo.trainCfg.get('gpu', False)
        params['trials'] = algo.trainCfg.get('trials', 1)
        params['timeout'] = algo.trainCfg.get('timeout')
        params['epochs'] = algo.trainCfg.get('epochs', 1)
        params['score'] = algo.trainCfg.get('score', None)
        params['threshold'] = algo.trainCfg.get('threshold', None)

        # early stop
        params['stop'] = dict()
        if params['timeout']:
            # timeout is minute per trial
            # convert to second for early stop
            # 'time_total_s' is fixed name of ray
            params['stop']['time_total_s'] = params['timeout'] * 60
        if params['score'] and params['threshold']:
            # These metrics are assumed to be increasing
            params['stop'][params['score']] = params['threshold']

        # tune parameters
        tuner['epochs'] = algo.trainCfg.get('epochs', 1)

        # search space based on parameters
        if algo.trainCfg.get('params'):
            params['args']: list = []
            search_space = algo.trainCfg.get('params')
            for key, value in search_space.items():
                params['args'].append(key)
                if value.startswith('('):
                    # (start, stop, step)
                    # ex: (2, 10, 2)
                    value_tmp = value.replace('(', '[').replace(')', ']')
                    values = execjs.eval(value_tmp)
                    if len(values) > 2:
                        # ex: (2, 10, 2)
                        if '.' in value:
                            # convert to float if one of value is float
                            tuner[key] = ray.tune.quniform(values[0], values[1], values[2])
                        else:
                            # convert to integer
                            tuner[key] = ray.tune.qrandint(values[0], values[1], values[2])
                    elif len(values) > 1:
                        # ex: (2.8, 9.5)
                        if '.' in value:
                            # convert to float if one of value is float
                            tuner[key] = ray.tune.uniform(values[0], values[1])
                        else:
                            tuner[key] = ray.tune.randint(values[0], values[1])
                    elif len(values) > 0:
                        tuner[key] = values[0]
                elif value.startswith('['):
                    # ex: [linear, rbf]
                    values = execjs.eval(value)
                    if len(values) > 0:
                        tuner[key] = ray.tune.choice(values)

                else:
                    # 8.5 or 'gini'
                    if '.' in value:
                        tuner[key] = float(value)
                    else:
                        tuner[key] = int(value) if value.isdigit() else value

    return params
