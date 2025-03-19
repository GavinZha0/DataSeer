import sys
import operator
import re
from datetime import datetime
import importlib
import execjs
import ray
import mlflow
import torch
import torchmetrics
from sqlalchemy.ext.asyncio import AsyncSession
from torch import nn

from apps.ml.algo.algo_trainable import build_ray_trainable
from config import settings
from core.crud import RET
from apps.datamgr import crud as dm_crud
from apps.ml import crud as ml_crud, schema as ml_schema
import sklearn
from sklearn.utils import all_estimators
import sklearn.datasets as skds
import xgboost as xgb
import lightgbm
import lightning
from utils.ray.ray_pipeline import RayPipeline
import inspect
import ast

som_doc = '''
If your dataset has 150 samples, 5*sqrt(150) = 61.23
hence a map 8-by-8 should perform well.

    Parameters
    ----------
    x : int
        x dimension of the SOM.

    y : int
        y dimension of the SOM.

    input_len : int
        Number of the elements of the vectors in input.

    sigma : float, optional (default=1)
        Spread of the neighborhood function.

        Needs to be adequate to the dimensions of the map
        and the neighborhood function. In some cases it
        helps to set sigma as sqrt(x^2 +y^2).

    learning_rate : float, optional (default=0.5)
        Initial learning rate.

        Adequate values are dependent on the data used for training.

        By default, at the iteration t, we have:
            learning_rate(t) = learning_rate / (1 + t * (100 / max_iter))

    decay_function : string or callable, optional
    (default='inverse_decay_to_zero')
        Function that reduces learning_rate at each iteration.
        Possible values: 'inverse_decay_to_zero', 'linear_decay_to_zero',
                         'asymptotic_decay' or callable

        If a custom decay function using a callable
        it will need to to take in input
        three parameters in the following order:

        1. learning rate
        2. current iteration
        3. maximum number of iterations allowed

        Note that if a lambda function is used to define the decay
        MiniSom will not be pickable anymore.

    neighborhood_function : string, optional (default='gaussian')
        Function that weights the neighborhood of a position in the map.
        Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

    topology : string, optional (default='rectangular')
        Topology of the map.
        Possible values: 'rectangular', 'hexagonal'

    activation_distance : string, callable optional (default='euclidean')
        Distance used to activate the map.
        Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'

        Example of callable that can be passed:

        def euclidean(x, w):
            return linalg.norm(subtract(x, w), axis=-1)

    random_seed : int, optional (default=None)
        Random seed to use.

    sigma_decay_function : string, optional
    (default='inverse_decay_to_one')
        Function that reduces sigma at each iteration.
        Possible values: 'inverse_decay_to_one', 'linear_decay_to_one',
                         'asymptotic_decay'

        The default function is:
            sigma(t) = sigma / (1 + (t * (sigma - 1) / max_iter))
'''


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
    import torchvision
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
sklearn category: 'classifier', 'regressor', 'transformer', 'cluster'
pytorch category: vision, audio, text
boost category: xgboost, lightgbm, catboost
"""
async def extract_algos(category: str):
    # e.g., category = 'sklearn.classifier' or 'boost.xgboost'
    cat = category.upper()

    result = {}
    if cat.startswith('SKLEARN'):
        temp_cat = category.split('.')
        sub_cat = temp_cat[len(temp_cat) - 1]
        estimators = all_estimators(type_filter=sub_cat)
        for name, class_ in estimators:
            m_name = class_.__dict__.get('__module__').split(".")[1]
            cls_name = class_.__name__
            if m_name != 'dummy':
                if result.get(m_name) is None:
                    result[m_name] = [cls_name]
                else:
                    result[m_name].append(cls_name)
    elif cat.startswith('PYTORCH'):
        temp_cat = category.split('.')
        sub_cat = temp_cat[len(temp_cat) - 1]
        if cat.endswith('CUSTOM'):
            result['root'] = ['CNN', 'GRU', 'LSTM', 'MLP', 'RNN']
        else:
            models = torch.hub.list(f'pytorch/{sub_cat}')
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
            result['misc'] = []
            [result['misc'].append(k) for k in add_list]
    elif cat.startswith('ANN'):
        result['root'] = []
        result['root'].append('MiniSOM')
    elif cat.startswith('BOOST'):
        if cat.endswith('XGBOOST'):
            xgb_models = xgb.__all__
            for k in xgb_models:
                if k == 'XGBClassifier' or k == 'XGBRFClassifier':
                    result[k] = ['binary:logistic', 'binary:logitraw', 'binary:hinge', 'multi:softmax', 'multi:softprob']
                elif k == 'XGBRegressor' or k == 'XGBRFRegressor':
                    result[k] = ['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror', 'reg:absoluteerror', 'reg:quantileerror', 'reg:gamma', 'reg:tweedie', 'count:poisson']
                elif k == 'XGBRanker':
                    result[k] = ['rank:pairwise', 'rank:ndcg', 'rank:map']
        elif cat.endswith('LIGHTGBM'):
            lgb_models = lightgbm.__all__
            for k in lgb_models:
                if k == 'LGBMClassifier':
                    result[k] = ['binary', 'multiclass', 'multiclassova', 'num_class']
                elif k == 'LGBMRegressor':
                    result[k] = ['regression_l1', 'regression_l2', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma', 'tweedie']
                elif k == 'LGBMRanker':
                    result[k] = ['lambdarank', 'rank_xendcg', 'rank_xendcg']
        elif cat.endswith('CATBOOST'):
            # catboost 1.2.7 requires numpy<2.0,>=1.16.0, but you have numpy 2.0.2 which is incompatible.
            result['root'] = []
            result['root'].append('CatBoostClassifier')
            result['root'].append('CatBoostRegressor')

    # sort result by key and ignore case
    result_json = dict(sorted(result.items(), key=lambda x: x[0].lower()))
    return result_json


"""
extract arguments of algorithm
sklearn category: 'classifier', 'regressor', 'transformer', 'cluster'
pytorch category: vision, audio, text
boost category: xgboost, lightgbm, catboost
"""
async def extract_algo_args(category: str, algo: str):
    # e.g., category = 'sklearn.classifier' or 'boost.xgboost'
    # e.g., algo = 'linear_model.LogisticRegression' or 'XGBClassifier'
    cat = category.upper()

    temp_algo = algo.split('.')
    group = temp_algo[0]
    algorithm = temp_algo[len(temp_algo)-1]

    args = []
    algo_func = None
    algo_doc = None
    if cat.startswith('SKLEARN'):
        # find algo func from sklearn estimators
        # can be ignored arguments
        temp_cat = category.split('.')
        sub_cat = temp_cat[len(temp_cat) - 1]
        ignored = ['estimator', 'verbose', 'n_jobs', 'random_state', 'max_iter']
        estimators = all_estimators(type_filter=sub_cat)
        algo_funcs = [class_ for name, class_ in estimators if name == algorithm]
        algo_func = algo_funcs[0]
        if algo_func:
            # cut doc to get parameter description
            algo_doc = algo_func.__doc__
            para_idx = algo_doc.find('------\n')
            if para_idx >= 0:
                # ex: 'Parameters\n ----------\n'
                algo_doc = algo_doc[para_idx + 8:]

            para_idx = algo_doc.find('------\n')
            if para_idx > 0:
                # ex: 'Attributes\n --------\n'
                algo_doc = algo_doc[:para_idx - 20]
    elif cat.startswith('PYTORCH'):
        if cat.endswith('CUSTOM'):
            if algorithm in ['RNN', 'GRU', 'LSTM']:
                algo_func = getattr(nn, algorithm)
                if algo_func:
                    # cut doc to get parameter description
                    algo_doc = algo_func.__doc__
                    para_idx = algo_doc.find('Args:\n')
                    if para_idx >= 0:
                        # ex: 'Args:\n'
                        algo_doc = algo_doc[para_idx + 5:]

                    para_idx = algo_doc.find('Attributes:\n')
                    if para_idx > 0:
                        algo_doc = algo_doc[:para_idx - 4]
            arguments = {}
            arguments['MLP']=[dict(name='input_size', default=1), dict(name='output_size', default=1)]
            arguments['RNN'] = [dict(name='input_size', default=1), dict(name='output_size', default=1),
                                dict(name='hidden_size', default=8), dict(name='num_layers', default=1),
                                dict(name='bias', default=True), dict(name='batch_first', default=False),
                                dict(name='dropout', default=0.0), dict(name='bidirectional', default=False),
                                dict(name='nonlinearity', default='tanh')]
            arguments['GRU'] = [dict(name='input_size', default=1), dict(name='output_size', default=1),
                                dict(name='num_layers', default=1), dict(name='bias', default=True),
                                dict(name='batch_first', default=False), dict(name='dropout', default=0.0),
                                dict(name='bidirectional', default=False)]
            arguments['LSTM'] = [dict(name='input_size', default=1), dict(name='hidden_size', default=8),
                    dict(name='num_layers', default=1), dict(name='bias', default=True),
                    dict(name='batch_first', default=False), dict(name='dropout', default=0.0),
                    dict(name='bidirectional', default=False), dict(name='proj_size', default=0),
                    dict(name='seq_len', default=1), dict(name='gap_len', default=1),
                    dict(name='output_size', default=1)]
            return dict(algo=algo, args=arguments[algorithm], doc=algo_doc)
        else:
            import torchvision
            # can be ignored arguments
            ignored = ['kwargs']
            algo_func = torchvision.models.get_model_builder(algorithm)
            if algo_func:
                # cut doc to get parameter description
                algo_doc = algo_func.__doc__
                para_idx = algo_doc.find('Args:\n')
                if para_idx >= 0:
                    # ex: 'Args:\n'
                    algo_doc = algo_doc[para_idx + 6:]

                para_idx = algo_doc.find('.. ')
                if para_idx > 0:
                    # ex: '.. autoclass::'
                    algo_doc = algo_doc[:para_idx - 4]
    elif cat.startswith('BOOST'):
        # ignored arguments
        # objective is in algoName (XGBClassifier.multi:softmax)
        # eval_metrics is in trainCfg (metrics='logloss')
        # device is in trainCfg (gpu=True)
        ignored = ['kwargs', 'verbosity', 'n_jobs', 'nthread', 'silent', 'objective', 'eval_metric', 'device', 'validate_parameters', 'interaction_constraints']
        if cat.endswith('XGBOOST'):
            algo_func = getattr(xgb, group)
            if algo_func:
                # cut doc to get parameter description
                algo_doc = algo_func.__doc__
                para_idx = algo_doc.find('Parameters\n')
                if para_idx >= 0:
                    # ex: 'Parameters\n'
                    algo_doc = algo_doc[para_idx + 11:]

                para_idx = algo_doc.find('Attributes\n')
                if para_idx > 0:
                    # ex: 'Attributes\n'
                    algo_doc = algo_doc[:para_idx - 1]
        elif cat.endswith('LIGHTGBM'):
            algo_func = getattr(lightgbm, group)
            if algo_func:
                # cut doc to get parameter description
                algo_doc = algo_func._base_doc
                para_idx = algo_doc.find('Parameters\n')
                if para_idx >= 0:
                    # ex: 'Parameters\n'
                    algo_doc = algo_doc[para_idx + 11:]

                para_idx = algo_doc.find('Returns\n')
                if para_idx > 0:
                    # ex: 'Attributes\n'
                    algo_doc = algo_doc[:para_idx - 1]
        elif cat.endswith('CATBOOST'):
            # to do
            return []
    elif cat.startswith('ANN'):
        if 'SOM' in algorithm:
            arguments = {}
            arguments['SOM'] = [dict(name='x', default=8), dict(name='y', default=8),
                                dict(name='input_len', default=1), dict(name='sigma', default=1),
                                dict(name='learning_rate', default=0.5), dict(name='decay_function', default='asymptotic_decay'),
                                dict(name='neighborhood_function', default='gaussian'), dict(name='topology', default='rectangular'),
                                dict(name='activation_distance', default='euclidean'), dict(name='random_seed', default=42),
                                dict(name='sigma_decay_function', default='asymptotic_decay')]
            return dict(algo=algo, args=arguments['SOM'], doc=som_doc)
    else:
        return []

    if algo_func:
        if cat.endswith('XGBOOST'):
            params = algo_func().get_xgb_params()
            args = [dict(name=key, default=params[key]) for key in params if key not in ignored]
        elif cat.endswith('LIGHTGBM'):
            params = algo_func().get_params()
            args = [dict(name=key, default=params[key]) for key in params if key not in ignored]
        else:
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
extract metrics of algorithm
sklearn category: 'classifier', 'regressor', 'transformer', 'cluster'
pytorch category: vision, audio, text
boost category: xgboost, lightgbm, catboost
"""
async def extract_algo_metrics(category: str, algo: str):
    # silhouette_score, calinski_harabasz_score and davies_bouldin_score can't get from get_scorer_names()
    # they don't need target value to evaluate cluster. all others need.
    # e.g., category = 'sklearn.classifier' or 'boost.xgboost'
    # e.g., algo = 'linear_model.LogisticRegression' or 'XGBClassifier'
    cat = category.upper()
    temp_algo = algo.split('.')
    group = temp_algo[0]
    algorithm = temp_algo[len(temp_algo) - 1]

    metrics = dict(clf=[], reg=[], cluster=[])
    if cat.startswith('SKLEARN'):
        temp_cat = category.split('.')
        sub_cat = temp_cat[len(temp_cat) - 1]
        names = sklearn.metrics.get_scorer_names()
        metrics['cluster'] = [it for it in names if
                             'rand_' in it or 'info_' in it or '_measure_' in it or 'homogeneity' in it
                             or 'fowlkes' in it or 'completeness' in it]
        metrics['cluster'].insert(0, 'davies_bouldin_score')
        metrics['cluster'].insert(0, 'calinski_harabasz_score')
        metrics['cluster'].insert(0, 'silhouette_score')
        metrics['regressor'] = [it for it in names if
                         '_mean_' in it or '_variance' in it or '_error' in it or 'r2' in it or 'd2_' in it]
        metrics['classifier'] = [it for it in names if it not in metrics['cluster'] and it not in metrics['regressor']]
        return metrics[sub_cat]
    elif cat.startswith('BOOST'):
        if cat.endswith('LIGHTGBM'):
            if 'Classifier' in group:
                # LogLoss: Logistic Loss, Negative Log-Likelihood or Cross-Entropy Loss(负对数损失), mlogloss: Multiclass Log Loss
                # auc: Area under the Curve, aucpr: Area Under the Precision-Recall Curve, map: Mean Average Precision(平均精确度)
                if 'binary' in algorithm:
                    return ['binary', 'binary_error', 'auc']
                else: # 'multi'
                    return ['multi_logloss', 'multi_error', 'auc']
            elif 'Regressor' in group:
                # MAE: Mean Absolute Error(平均绝对值误差), MPE: Mean Percentage Error(平均百分比误差), MSPE:Mean Squared Prediction Error(均方百分比误差),
                # MAPE: Mean Absolute Pencentage Error(平均绝对百分比误差), MPHE: Mean Pseudo Huber Error
                # MSE: Mean Squared Error(均方误差), RMSE: Root Mean Squared Error(均方根误差), RMSLE: Root Mean Squared Logarithmic Error(均方根对数误差)
                # NDCG: Normalized Discounted Cumulative Gain, MAP: Mean Average Precision
                return ['l1', 'l2', 'rmse', 'quantile', 'mape', 'huber', 'fair', 'poisson', 'gamma', 'gamma_deviance', 'tweedie']
            elif 'Ranker' in group:
                # pre: Precision at k, ndcg: Normalized Discounted Cumulative Gain(归一化折损累计增益)
                return ['auc', 'pre', 'ndcg' 'map']
        if cat.endswith('XGBOOST'):
            # sklearn.metrics can be used by xgboost when use sklearn API
            # all xgboost metrics should be covert to lower case when they are applied
            if 'Classifier' in group:
                # LogLoss: Logistic Loss, Negative Log-Likelihood or Cross-Entropy Loss(负对数损失), mlogloss: Multiclass Log Loss
                # auc: Area under the Curve, aucpr: Area Under the Precision-Recall Curve, map: Mean Average Precision(平均精确度)
                if 'binary' in algorithm:
                    return ['logloss', 'error', 'auc', 'acupr', 'map']
                else: # 'multi'
                    return ['merror', 'mlogloss', 'auc', 'aucpr', 'map']
            elif 'Regressor' in group:
                # MAE: Mean Absolute Error(平均绝对值误差), MPE: Mean Percentage Error(平均百分比误差), MSPE:Mean Squared Prediction Error(均方百分比误差),
                # MAPE: Mean Absolute Pencentage Error(平均绝对百分比误差), MPHE: Mean Pseudo Huber Error
                # MSE: Mean Squared Error(均方误差), RMSE: Root Mean Squared Error(均方根误差), RMSLE: Root Mean Squared Logarithmic Error(均方根对数误差)
                # NDCG: Normalized Discounted Cumulative Gain, MAP: Mean Average Precision
                return ['mae', 'mape', 'mphe', 'rmse', 'rmsle', 'logloss', 'poisson-nloglik', 'gamma-nloglik', 'gamma-deviance', 'cox-nloglik', 'tweedie-nloglik']
            elif 'Ranker' in group:
                # pre: Precision at k, ndcg: Normalized Discounted Cumulative Gain(归一化折损累计增益)
                return ['auc', 'pre', 'ndcg' 'map']
    elif cat.startswith('PYTORCH'):
        if cat.endswith('IMAGE'):
            return [torchmetrics.image.__all__]
        elif cat.endswith('AUDIO'):
            return [torchmetrics.audio.__all__]
        elif cat.endswith('TEXT'):
            return [torchmetrics.text.__all__]
        elif cat.endswith('CLASSIFICATION'):
            return [torchmetrics.classification.__all__]
        elif cat.endswith('REGRESSION'):
            return [torchmetrics.regression.__all__]
        elif cat.endswith('DETECTION'):
            return [torchmetrics.detection.__all__]
        else:
            return nn.modules.loss.__all__
    else:
        return []


"""
build train pipeline
"""


async def train_pipeline(algo_id: int, db: AsyncSession, user: dict):
    # get algo, dataset and datasource info from db
    algo_info = await ml_crud.AlgoDal(db).get_data(algo_id, v_ret=RET.SCHEMA)
    dataset_info = await ml_crud.DatasetDal(db).get_data(algo_info.dataCfg.get('datasetId'), v_ret=RET.SCHEMA)
    source_info = await dm_crud.DatasourceDal(db).get_data(dataset_info.sourceId, v_ret=RET.SCHEMA)
    num_uniques = [field['nunique'] for field in dataset_info.fields if 'target' in field and 'nunique' in field]
    targets = [field['name'] for field in dataset_info.fields if 'target' in field and 'omit' not in field]

    # build parameters
    params = await build_params(algo_info, user)
    if num_uniques:
        # num_class for classifier of XGBoost/LightGBM
        # don't add it for regression
        if 'binary' in algo_info.algoName or 'multi' in algo_info.algoName:
            params['tune_param']['num_class'] = num_uniques[0]

    if targets:
        # target names
        params['tune_param']['targets'] = targets

    ready = await build_ray_trainable(algo_info.category, algo_info.algoName, algo_info.srcCode, params)
    if ready is False:
        return False

    # import module from saved file
    dy_module = importlib.import_module(f'temp.ml.{params["module_name"]}')

    # detect classes
    train_func = None
    dy_cls = [cls for name, cls in inspect.getmembers(dy_module, inspect.isclass) if params["module_name"] in cls.__module__]
    if dy_cls is None or len(dy_cls) == 0:
        return False
    elif dy_cls and len(dy_cls) > 0:
        for cls in dy_cls:
            # must have a function 'train'
            dy_func = [func for name, func in inspect.getmembers(cls, inspect.isfunction) if name in ['train'] and cls.__name__ in func.__qualname__]
            if dy_func and len(dy_func) > 0:
                train_func = dy_func[0]
                break

    # return if no trainable function for Ray
    if train_func is None:
        return False

    # initialize RayPipeline and run training
    pipeline = RayPipeline.remote(source_info, dataset_info, algo_info)
    # run: load -> transform -> train
    pipeline.run.remote(train_func, params)



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

    if algo.dataCfg:
        tuner['batch_size'] = algo.dataCfg.get('batchSize', 1)

    if algo.trainCfg:
        params['gpu'] = algo.trainCfg.get('gpu', False)
        params['trials'] = algo.trainCfg.get('trials', 1)
        params['epochs'] = algo.trainCfg.get('epochs', 1)
        params['timeout'] = algo.trainCfg.get('timeout')
        params['metrics'] = algo.trainCfg.get('metrics', None)
        params['threshold'] = algo.trainCfg.get('threshold', None)
        # tune parameters
        tuner['epochs'] = params['epochs']
        # for xgboost
        tuner['device'] = 'cuda' if params['gpu'] else 'cpu'
        tuner['eval_metric'] = params['metrics']
        # for lightgbm
        tuner['metric'] = params['metrics']

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
