import base64
from datetime import datetime
import importlib
import pandas as pd
import ray
from pandas import CategoricalDtype
from sqlalchemy.ext.asyncio import AsyncSession
from config import settings
from core.crud import RET
from utils.db_executor import DbExecutor
from apps.datamgr import crud as dm_crud
from apps.ml import crud as ml_crud, schema as ml_schema
from sklearn import preprocessing as pp
from sklearn import feature_extraction as fe
from utils.ray_tuner import RayTuner

SETUP_MLFLOW_IN_TRAIN_FUNC = '''
    mlflow.set_tracking_uri(config.get('tracking_url'))
    mlflow.set_experiment(experiment_id=config.get('exper_id'))
    mlflow.autolog(extra_tags={MLFLOW_RUN_NAME: ray.train.get_context().get_trial_id(), MLFLOW_USER: config.get('user_id')})
'''

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
    dataframe, total = await db.db_query(dataset_info.query, None, dataset_info.variable)
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
                df[field_name] = pp.KBinsDiscretizer(n_bins=10, strategy='uniform', encode='ordinal').fit_transform(df[field_name])
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
extract existing algorithms
framework: sklearn, pytorch, tensorflow
sklearn type: 'classifier', 'regressor', 'transformer', 'cluster'
pytorch type: 
"""
async def extract_existing_algos(framework: str, category: str):
    category_map = {'clf': 'classifier', 'reg': 'regressor', 'cluster': 'cluster', 'transform': 'transformer'}
    result_tree = []
    result = {}
    match framework:
        case 'sklearn':
            from sklearn.utils import all_estimators
            estimators = all_estimators(type_filter=category_map[category])
            for name, class_ in estimators:
                module_name = class_.__dict__.get('__module__').split(".")[1]
                class_name = class_.__name__
                if result.get(module_name) is None:
                    result[module_name] = [class_name]
                else:
                    result[module_name].append(class_name)

            for module_name in result:
                result_tree.append({'name': module_name, 'children': []})
                for class_name in result[module_name]:
                    result_tree[-1]['children'].append({'name': class_name})
        case 'pytorch':
            from sklearn.utils import all_estimators
            estimators = all_estimators(type_filter='classifier')

    return result_tree


"""
return pandas dataframe
"""
async def train_pipeline(algo_id: int, db: AsyncSession, user: dict):
    # get algo, dataset and datasource info from db
    algo_info = await ml_crud.AlgoDal(db).get_data(algo_id, v_ret=RET.SCHEMA)
    dataset_info = await ml_crud.DatasetDal(db).get_data(algo_info.datasetId, v_ret=RET.SCHEMA)
    source_info = await dm_crud.DatasourceDal(db).get_data(dataset_info.sourceId, v_ret=RET.SCHEMA)
    psw = base64.b64decode(source_info.password).decode('utf-8')
    targets = [field['name'] for field in dataset_info.fields if 'target' in field and 'omit' not in field]

    # build parameters
    params = await build_params(algo_info, user)

    # save algo content to local file
    train_code = algo_info.srcCode.replace('setup_mlflow()', SETUP_MLFLOW_IN_TRAIN_FUNC)
    with open(f'./temp/{params["module_name"]}.py', 'w') as file:
        file.write(train_code)

    # import module and class from saved file
    dy_module = importlib.import_module(f'temp.{params["module_name"]}')
    dy_cls = getattr(dy_module, params['cls_name'])

    if not ray.is_initialized():
        ray.init(local_mode=settings.RAY_LOCAL_MODE, ignore_reinit_error=True)

    # initialize RayUtils to create database connection
    rayTuner = RayTuner.remote(source_info.type, source_info.url, source_info.username, psw)
    # get dataset from datasource
    dataset_df = rayTuner.extract.remote(dataset_info.query)
    # transform data based on field config of dataset
    transformed_df = rayTuner.transform.remote(dataset_df, dataset_info.fields)
    # split and shuffle for training preparation
    train_test_data = rayTuner.shuffle.remote(transformed_df, targets, algo_info.attr['testRatio'])
    rayTuner.train.remote(algo_info.framework, params, dy_cls, train_test_data)


"""
train algorithm
"""
async def build_params(algo: ml_schema.Algo, user: dict):
    now_ts = int(datetime.now().timestamp())

    # get class name from srcCode
    cls_idx = algo.srcCode.index('class')
    cls_name = algo.srcCode[cls_idx + 5:]
    cls_idx = min(cls_name.index(':'), cls_name.index('('))
    cls_name = cls_name[:cls_idx].strip()

    # basic parameters
    # use mysql db as tracking store
    # use AWS S3 as file store
    # an experiment binds to an algo, experiment name is unique
    # every run has user info
    params: dict = {'algo_id': algo.id,
                    'algo_name': algo.name,
                    'user_id': user.id,
                    'user_name': user.name,
                    'org_id': user.oid,
                    'module_name': f"ml_algo_{algo.orgId}_{algo.id}",
                    'cls_name': cls_name,
                    'tracking_url': settings.SQLALCHEMY_MLFLOW_DB_URL,
                    's3_url': settings.AWS_S3_ENDPOINT,
                    's3_id': settings.AWS_S3_ACCESS_KEY,
                    's3_key': settings.AWS_S3_SECRET_KEY,
                    'artifact_location': f"s3://pie-org-{algo.orgId}/ml/ml_algo_{algo.orgId}_{algo.id}",
                    'exper_name': f"ml_algo_{algo.orgId}_{algo.id}",
                    'start_ts': now_ts,
                    'tune_param': {'user_id': user.id, 'epochs': 1, 'tracking_url': settings.SQLALCHEMY_MLFLOW_DB_URL}
                    }

    # tune parameters
    tuner = params['tune_param']
    if algo.config:
        params['trials'] = algo.config.get('trials', 1)
        params['timeout'] = algo.config.get('timeout')
        tuner['epochs'] = algo.config.get('epochs', 1)

    # search parameters
    if algo.attr and algo.attr.get('params'):
        params['args']: list = []
        for search_param in algo.attr.get('params'):
            params['args'].append(search_param['name'])
            if search_param['value'].startswith('('):
                value = search_param['value'][1:-1]
                values = value.split(',')
                if len(values) > 2:
                    if '.' in value:
                        # convert to float if one of value is float
                        tuner[search_param['name']] = ray.tune.quniform(float(values[0]), float(values[1]),
                                                                         float(values[2]))
                    else:
                        # convert to integer
                        tuner[search_param['name']] = ray.tune.qrandint(int(values[0]), int(values[1]),
                                                                         int(values[2]))
                elif values.length > 1:
                    if '.' in value:
                        tuner[search_param['name']] = ray.tune.uniform(float(values[0]), float(values[1]))
                    else:
                        tuner[search_param['name']] = ray.tune.randint(int(values[0]), int(values[1]))
                elif values.length > 0:
                    if '.' in value:
                        tuner[search_param['name']] = float(values[0])
                    else:
                        tuner[search_param['name']] = int(values[0])
            elif search_param['value'].startswith('['):
                value = search_param['value'][1:-1]
                values = value.split(',')
                if len(values) > 0:
                    if '.' in value:
                        converted_values = [float(val) if val.isdigit() else val for val in values]
                    else:
                        converted_values = [int(val) if val.isdigit() else val for val in values]
                    tuner[search_param['name']] = ray.tune.choice(converted_values)

    # early stop based on metrics
    if algo.config and algo.config.get('metrics'):
        params['metrics']: list = []
        early_stop = {}
        for eval_kpi in algo.config.get('metrics'):
            params['metrics'].append(eval_kpi['name'])
            if eval_kpi.get('value') is not None:
                early_stop[eval_kpi['name']] = float(eval_kpi['value'])
        if len(early_stop) > 0:
            params['stop'] = early_stop

    return params

