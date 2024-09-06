#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/15
# @File           : ray_tuner.py
# @desc           : train ml
import os
from typing import Dict
import mlflow
import pandas as pd
import ray
import ray.tune.search as search
import ray.tune.schedulers as schedule
from pandas import CategoricalDtype
from ray import tune, train
from ray.exceptions import RayError
from ray.tune.experimental.output import get_air_verbosity, AirVerbosity
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sklearn import preprocessing as pp
from sklearn import feature_extraction as fe
from config.settings import TEMP_DIR
from utils.ray.ray_reporter import RayReport, JOB_PROGRESS_START, JOB_PROGRESS_END


@ray.remote
class SklearnTrainer:
    def __init__(self, type: str, url: str, username: str, password: str):
        self.type = type
        self.engine = None
        self.dataset_df = None
        self.transformed_df = None
        self.train_data = None

        if type.upper() == 'MYSQL':
            self.engine = create_engine(f'mysql+mysqldb://{username}:{password}@{url}?charset=utf8mb4', echo=False)

    # extract data from db or file system
    def extract(self, sql: str, params: str = None):
        df = pd.read_sql(sql, self.engine)
        self.dataset_df = df
        return df

    # transform data based on dataset field config
    def transform(self, df: pd.DataFrame, fields, ratio: int, shuffle: bool):
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
        self.transformed_df = df

        targets = [field['name'] for field in fields if 'target' in field and 'omit' not in field]
        # Split the data into train and test sets.
        cols = df.columns.tolist()
        features = list(set(cols).difference(set(targets)))
        x, tx, y, ty = train_test_split(df[features], df[targets], test_size=ratio, shuffle=shuffle)
        data: dict = {'x': x, 'y': y.to_numpy().ravel(), 'tx': tx, 'ty': ty.to_numpy().ravel()}
        self.train_data = data
        return data

    # train ML algo based on ray and mlflow
    def train(self, params: dict, cls, data: Dict):
        # use AWS S3/minio as artifact repository
        os.environ["AWS_ACCESS_KEY_ID"] = params.get('s3_id')
        os.environ["AWS_SECRET_ACCESS_KEY"] = params.get('s3_key')
        mlflow.environment_variables.MLFLOW_S3_ENDPOINT_URL = params.get('s3_url')
        # use mysql db as tracking store
        mlflow.set_tracking_uri(params['tracking_url'])
        # resolve warning 'Experiment state snapshotting has been triggered multiple times in the last 5.0 seconds'
        os.environ['TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S'] = '0'
        # to disable log deduplication
        os.environ["RAY_DEDUP_LOGS"] = "0"
        # enable custom ProgressReporter when set to 0
        os.environ['RAY_AIR_NEW_OUTPUT'] = '0'

        # create a new experiment with UNIQUE name for mlflow (ex: algo_3_1234567890)
        exper_tags = dict(org_id=params['org_id'], algo_id=params['algo_id'], algo_name=params['algo_name'],
                          user_id=params['user_id'])
        if params.get('args'):
            exper_tags['args'] = '|'.join(params['args'])

        params['exper_id'] = mlflow.create_experiment(name=params['exper_name'], tags=exper_tags,
                                                      artifact_location=params['artifact_location'])
        params['tune_param']['exper_id'] = params['exper_id']
        # resolve the warning 'Matplotlib GUI outside of the main thread will likely fail'
        # matplotlib.use('agg')
        # mlflow.autolog()

        # create progress report
        progressRpt = RayReport(params)
        progressRpt.experimentProgress(JOB_PROGRESS_START)

        if params.get('gpu') == True:
            tune_func = tune.with_resources(tune.with_parameters(cls.train, data=data), resources={"gpu": 1})
        else:
            tune_func = tune.with_parameters(cls.train, data=data)

        tune_cfg = tune.TuneConfig(num_samples=params['trials'],
                                   search_alg=search.BasicVariantGenerator(max_concurrent=1),
                                   scheduler=schedule.ASHAScheduler(mode="max"),
                                   time_budget_s=params['timeout'] * 60 * params['trials'] if params.get('timeout') else None)
        # ray will save tune results into storage_path with sub-folder exper_name
        # this is not used because we are using mlflow to save result on S3
        run_cfg = train.RunConfig(name=params['exper_name'],  stop=params.get('stop'),
                                  verbose=get_air_verbosity(AirVerbosity.DEFAULT),
                                  log_to_file=False, storage_path=TEMP_DIR+'/tune/',
                                  checkpoint_config=False,
                                  callbacks=[progressRpt])

        tuner = tune.Tuner(trainable=tune_func,
                           tune_config=tune_cfg,
                           run_config=run_cfg,
                           param_space=params['tune_param'])

        try:
            # start train......
            result = tuner.fit()
        except RayError as e:
            print(e)
            # report exception
            progressRpt.experimentException(e)
        else:
            # report progress
            progressRpt.experimentProgress(JOB_PROGRESS_END)