#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/2/15
# @desc           : train ml

import base64
import os
from typing import Any, Dict
import mlflow
import numpy as np
import pandas as pd
import ray
import ray.tune.search as search
import ray.tune.schedulers as schedule
import torch
from pycaret.time_series import TSForecastingExperiment
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.anomaly import AnomalyExperiment
from ray.tune.stopper import TrialPlateauStopper
from torchvision.transforms import v2 as ttv2
import torchaudio
import torchvision
from pandas import CategoricalDtype
from ray import tune, train
from ray.exceptions import RayError
from ray.tune.experimental.output import get_air_verbosity, AirVerbosity
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp
from sklearn import feature_extraction as fe
import xgboost as xgb
import mysql.connector as ct
from config import settings
from config.settings import TEMP_DIR, RAY_NUM_GPU
from utils.ray.ray_reporter import RayReport, JOB_PROGRESS_START, JOB_PROGRESS_END
from ray.air import ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer
from ray.train.xgboost import XGBoostTrainer
from ray.train.lightgbm import LightGBMTrainer
import s3fs
import duckdb


# This is very similar with data_loader.py
# It runs remotely on ray for async response and data load performance is lower than data_loader.py
# This is used for ML training. So the data load performance difference is not most important.
# Both this and data_loader.py should keep the same logic and bug fix
@ray.remote
class RayWorkflow:
    def __init__(self, workflow_info: any):
        self.workflow = workflow_info

    def set_attr(self, name: str, attr: any):
        self.data_type = None
        self.transformed = False
        self.psw = None
        match name:
            case 'datasource':
                self.source_info = attr
                self.source_type = attr.type.upper()
                if attr.password and attr.password.upper() != 'NA' and attr.password != 'NONE':
                    # decode password
                    self.psw = base64.b64decode(attr.password).decode('utf-8')
                match self.source_type:
                    case 'MYSQL':
                        # get db info (host, port and db name)
                        self.host = attr.url.split(':')[0]
                        tmp = attr.url.split(':')[1]
                        self.port = int(tmp.split('/')[0])
                        self.db = tmp.split('/')[1]

                    case 'S3BUCKET':
                        # get bucket info (endpoint and bucket name)
                        https = attr.url.split('//')
                        urls = https[1].split('/')
                        self.bucket = urls[1]

                        # S3Fs is a Pythonic file interface to S3.
                        # it can mount a bucket as directory while preserving the native object format for files.
                        # A potential replacement of S3FS is JuiceFS.
                        self.fs = s3fs.S3FileSystem(use_ssl=False, client_kwargs={
                            "aws_access_key_id": settings.AWS_S3_ACCESS_ID,
                            "aws_secret_access_key": settings.AWS_S3_SECRET_KEY,
                            "endpoint_url": settings.AWS_S3_ENDPOINT,
                            "verify": False})
                    case 'BUILDIN':
                        # for pytorch, huggingface, etc.
                        nothing = None
            case 'dataset':
                self.dataset_info = attr
                self.data_type = attr.type.upper()
            case 'algorithm':
                self.algo_info = attr
                # category = 'sklearn.classifier'
                # category = 'boost.xgboost'
                self.algo_cat = attr.category.upper()
            case 'transformed':
                self.transformed = attr
            case 'data_type':
                self.data_type = attr
            case 'algo_cat':
                self.algo_cat= attr

    def load_data(self, kind: str, source_info: any, dataset_info: any) -> ray.data.Dataset:
        self.source_info = source_info
        self.source_type = source_info.type.upper()
        self.dataset_info = dataset_info
        self.data_type = dataset_info.type.upper()

        if source_info.password and source_info.password.upper() != 'NA' and source_info.password != 'NONE':
            # decode password
            self.psw = base64.b64decode(source_info.password).decode('utf-8')
        match self.source_type:
            case 'MYSQL':
                # get db info (host, port and db name)
                self.host = source_info.url.split(':')[0]
                tmp = source_info.url.split(':')[1]
                self.port = int(tmp.split('/')[0])
                self.db = tmp.split('/')[1]

            case 'S3BUCKET':
                # get bucket info (endpoint and bucket name)
                https = source_info.url.split('//')
                urls = https[1].split('/')
                self.bucket = urls[1]

                # S3Fs is a Pythonic file interface to S3.
                # it can mount a bucket as directory while preserving the native object format for files.
                # A potential replacement of S3FS is JuiceFS.
                self.fs = s3fs.S3FileSystem(use_ssl=False, client_kwargs={
                    "aws_access_key_id": settings.AWS_S3_ACCESS_ID,
                    "aws_secret_access_key": settings.AWS_S3_SECRET_KEY,
                    "endpoint_url": settings.AWS_S3_ENDPOINT,
                    "verify": False})
            case 'BUILDIN':
                # for pytorch, huggingface, etc.
                nothing = None

        ds: ray.data.Dataset = None
        # return ray dataset
        match self.source_type:
            case 'MYSQL':
                # return ray.data.Dataset
                ds = self.load_from_db(dataset_info.content, dataset_info.variable)
            case 'S3BUCKET':
                # return ray.data.Dataset
                ds = self.load_from_bucket(dataset_info.content)
            case 'BUILDIN':
                # return ray.data.Dataset
                ds = self.load_from_buildin(dataset_info.content)
        print(f'workflow>>>load_data: {ds.count()}')
        return ds


    def load_from_db(self, content: str, params: []) -> ray.data.Dataset:
        # serialization issue of ray.data.read_sql is resolved by this way
        def create_conn(usr: str, psw: str, host: str, db: str):
            def conn():
                # worked with mysql-connector-python 9.0.0
                # failed with mysql-connector-python 9.1.0
                return ct.connect(user=usr, password=psw, host=host, database=db)
            return conn

        conn_factory = create_conn(self.source_info.username, self.psw, self.host, self.db)
        ds = ray.data.read_sql(content, conn_factory, override_num_blocks=1)
        # Dataset(num_rows=123,schema={sepal_length: double,sepal_width: double,petal_length: double,petal_width: double,target: int64})
        # return ray dataset
        return ds


    def load_from_bucket(self, content: str) -> ray.data.Dataset:
        if self.data_type == 'DATA':
            # find files from content and get file list
            # ex: SELECT * FROM 'mldata/iris.csv'
            file_list = [f.strip("'") for f in content.split(' ') if
                         f.upper().endswith((".CSV'", ".JSON'", ".PARQUET'", ".TXT'"))]
            # get unique files
            file_list = list(set(file_list))
            duck_sql = content
            # support 3 files only in content (usually there is 1 file)
            df0 = df1 = df2 = None

            # load files from s3 bucket using ray functions
            for idx, f_name in enumerate(file_list):
                # # parameter for zip file: arrow_open_stream_args = {"compression": "gzip"}
                if f_name.upper().endswith(('.CSV')):
                    # get data from csv file using ray
                    ds = ray.data.read_csv(filesystem=self.fs, paths=f"s3://{self.bucket}/{f_name}")
                elif f_name.upper().endswith(('.JSON')):
                    # get data from json file using ray
                    ds = ray.data.read_json(filesystem=self.fs, paths=f"s3://{self.bucket}/{f_name}")
                elif f_name.upper().endswith(('.PARQUET')):
                    # get data from parquet file using ray
                    ds = ray.data.read_parquet(filesystem=self.fs, paths=f"s3://{self.bucket}/{f_name}")
                elif f_name.upper().endswith(('.TXT')):
                    # get data from text file using ray
                    ds = ray.data.read_text(filesystem=self.fs, paths=f"s3://{self.bucket}/{f_name}")
                else:
                    continue

                df = ds.to_pandas()
                # support 3 files only in content (usually there is 1 file)
                match idx:
                    case 0:
                        df0 = df
                    case 1:
                        df1 = df
                    case 2:
                        df2 = df
                    case '_':
                        continue

                # replace file name with dfX
                duck_sql = duck_sql.replace(f_name, f'df{idx}')
            # duckDB supports to query data from files using sql syntax
            # run sql to get data then convert to dataframe
            df = duckdb.sql(duck_sql).df()
            # convert to ray dataset
            return ray.data.from_pandas(df)
        elif self.data_type == 'IMAGE':
            # find folders from content and get folder list
            # ex: SELECT * FROM 'mldata/images/'
            folder_list = [f.strip("'") for f in content.split(' ') if f.lower().endswith(("/'"))]
            # get unique files
            folder_list = list(set(folder_list))
            if folder_list:
                f_name = folder_list[0]
                ds = ray.data.read_images(filesystem=self.fs, paths=f"s3://{self.bucket}/{f_name}")
                return ds
            else:
                return None


    def load_from_buildin(self, content: str) -> ray.data.Dataset:
        if content.startswith('sklearn'):
            import sklearn
            from sklearn import datasets
            dataset = eval(content)()
            df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            if 'target' in dataset.keys():
                df['target'] = dataset.target
            return ray.data.from_pandas(df)
        elif content.startswith('torchvision'):
            import torchvision
            from torchvision import datasets
            dataset_func = eval(content)
            transforms = self.get_img_transforms(self.dataset_info.transform)
            # it will be downloaded if it doesn't exist in local folder
            # transform is applied when download dataset
            data: torchvision.datasets.MNIST = dataset_func(TEMP_DIR + '/data/', download=True, transform=transforms)
            # mark as transformed when loading image data
            self.transformed = True
            # ArrowConversionError: Error converting data to Arrow: [(tensor([[[
            return ray.data.from_torch(data)
        elif content.startswith('torchaudio'):
            import torchaudio
            from torchaudio import datasets
            dataset_func = eval(content)
            # it will be downloaded if it doesn't exist in local folder
            data = dataset_func(TEMP_DIR + '/data/', download=True)
            # convert to ray dataset
            ds = ray.data.from_torch(data)
            return ds
        elif content.startswith('tensorflow'):
            # tf_ds, _ = tfds.load("cifar10", split=["train", "test"])
            # ds = ray.data.from_tf(tf_ds)
            aaa = 111
        elif content.startswith('huggingface'):
            # hf_ds = load_dataset("wikitext", "wikitext-2-raw-v1")
            # ray_ds = ray.data.from_huggingface(hf_ds["train"])
            bbb = 222

    # transform data based on dataset field config
    def transform(self, kind: str, params: dict, ds: ray.data.Dataset):
        print(f'workflow>>>transform: {ds.count()}')
        trans_ds: ray.data.Dataset = None
        match self.data_type:
            case 'DATA' | 'TIMESERIES':
                # tabular data transformation based on RAY Dataset
                trans_ds = self.trans_data(kind, params, ds)
        print(f'workflow>>>>>>transform: {trans_ds.count()}')
        return trans_ds



    # transform tabular data based on field config
    def trans_data(self, kind: str, params: dict, dataset: ray.data.Dataset)->ray.data.Dataset:
        if dataset is None or dataset.count == 0:
            return None

        # get fields
        fields = self.dataset_info.fields
        omitFields = [field['name'] for field in fields if 'omit' in field]
        validFields = [field['name'] for field in fields if 'omit' not in field]
        targetFields = [field['name'] for field in fields if 'target' in field and 'omit' not in field]
        featureFields = list(set(validFields).difference(set(targetFields)))

        # remove omit fields
        common_fields = list(set(omitFields).intersection(set(dataset.columns())))
        if len(common_fields) > 0:
            ds = dataset.drop_columns(common_fields)
        else:
            ds = dataset

        # convert ray dataset to pandas dataframe
        # pandas has more methods to transform data
        # Polars is a potential replacement of pandas for data analysis. it is faster than pandas.
        df: pd.DataFrame = ds.to_pandas()

        # drop the row/column if all values are na
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        # drop all duplicate rows
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        valid_fields = [field for field in fields if 'omit' not in field]
        # data type conversion
        for it in valid_fields:
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
                # mean, most_frequent, constant (fill_value)
                # preprocessor = raypp.SimpleImputer(columns=[field_name], strategy="mean", fill_value=None)
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
                        df[[field_name]] = pp.OrdinalEncoder().fit_transform(df[[field_name]])
                case 'hot':  # One-Hot
                    df[[field_name]] = pp.OneHotEncoder().fit_transform(df[[field_name]])
                case 'hash':  # Hashing
                    df[[field_name]] = fe.FeatureHasher(input_type='string').fit_transform(df[[field_name]])
                case 'binary':  # Binary
                    df[[field_name]] = pp.Binarizer(threshold=1).fit_transform(df[[field_name]])
                case 'bins':  # Binning
                    df[[field_name]] = pp.KBinsDiscretizer(n_bins=10, strategy='uniform',
                                                         encode='ordinal').fit_transform(df[[field_name]])
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
                    df[[field_name]] = pp.MinMaxScaler().fit_transform(df[[field_name]])
                case 'maxabs':
                    # [-1, 1]
                    df[[field_name]] = pp.MaxAbsScaler().fit_transform(df[[field_name]])
                case 'robust':
                    df[[field_name]] = pp.RobustScaler().fit_transform(df[[field_name]])
                case 'l1':
                    df[[field_name]] = pp.Normalizer(norm='l1').fit_transform(df[[field_name]])
                case 'l2':
                    df[[field_name]] = pp.Normalizer(norm='l2').fit_transform(df[[field_name]])

        ray_ds = ray.data.from_pandas(df)
        return ray_ds


    # transform/split/shuffle
    def trans_image(self, ds)->dict:
        # build transforms based on config
        # some operations are for image, some are for tensor
        # image transform fist then tensor transform
        # between them there should be a v2.ToTensor()
        # to do. -- Gavin

        if self.transformed:
            dataset = ds
        else:
            transform = self.dataset_info.transform
            # ds = dataset.to_torch()
            pipe = [ttv2.PILToTensor(), ttv2.ToDtype(torch.float32, scale=True)]
            for it in transform:
                match it['operation']:
                    case 'crop':
                        # Crops the given image at the center
                        pipe.append(ttv2.RandomCrop(it['param']))
                    case 'resize':
                        # Resize the input image to the given size
                        pipe.append(ttv2.Resize(it['param']))
                    case 'rotate':
                        # Rotate the image by angle
                        pipe.append(ttv2.RandomRotation(it['param']))
                    case 'hflip':
                        # Horizontally flip the given image randomly with a given probability
                        pipe.append(ttv2.RandomHorizontalFlip(it['param']))
                    case 'vflip':
                        # Vertically flip the given image randomly with a given probability
                        pipe.append(ttv2.RandomVerticalFlip(it['param']))
                    case 'pad':
                        # Pad the given image on all sides with the given "pad" value
                        pipe.append(ttv2.Pad(it['param']))
                    case 'brightness':
                        # Adjust brightness of the given image by factor
                        pipe.append(ttv2.ColorJitter(brightness=(it['param'])))
                    case 'contrast':
                        # Adjust contrast of the given image by factor
                        pipe.append(ttv2.ColorJitter(contrast=(it['param'])))
                    case 'saturation':
                        # Adjust color saturation of the given image by factor
                        pipe.append(ttv2.ColorJitter(saturation=(it['param'])))
                    case 'hue':
                        # Adjust hue of the given image by factor
                        pipe.append(ttv2.ColorJitter(hue=(it['param'])))
                    case 'grayscale':
                        # Convert image to grayscale
                        pipe.append(ttv2.Grayscale(it['param']))
                    case 'blur':
                        # Blur the given image by a Gaussian filter of radius
                        pipe.append(ttv2.GaussianBlur(it['param']))
                    case 'sharpness':
                        # Adjust the sharpness of the given image by factor
                        pipe.append(ttv2.RandomAdjustSharpness(it['param']))
                    case 'normalize':
                        pipe.append(ttv2.Normalize(mean=eval(it['param']['mean']), std=eval(it['param']['std'])))


            # define transformer
            transforms = ttv2.Compose(pipe)
            ds = ray.get(ds)

            def trans_img(row: Dict[str, Any]) -> Dict[str, Any]:
                # row['item'] is tuple (image, label)
                image, label = row['item']
                # transform image and replace original item
                row["item"] = (transforms(image), label)
                return row

            def trans_batch_img(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                for row in batch['item']:
                    image, label = row
                    row = (transforms(image), label)
                return batch

            dataset = ds.map(trans_img)
            # dataset = ds.map_batches(trans_batch_img, batch_size=64)

        shuffle = self.algo_info.dataCfg.get('shuffle', False)
        ratio = self.algo_info.dataCfg.get('evalRatio', None)
        if ratio == 0:
            ratio = None
        train_ds, val_ds = dataset.train_test_split(test_size=ratio, shuffle=shuffle)
        ray_ds = dict(train=train_ds, validation=val_ds)
        return ray_ds

    # get image transforms based on config
    def get_img_transforms(self, transform: list):
        pipe = [ttv2.PILToTensor(), ttv2.ToDtype(torch.float32, scale=True)]
        for it in transform:
            match it['operation']:
                case 'crop':
                    # Crops the given image at the center
                    pipe.append(ttv2.RandomCrop(it['param']))
                case 'resize':
                    # Resize the input image to the given size
                    pipe.append(ttv2.Resize(it['param']))
                case 'rotate':
                    # Rotate the image by angle
                    pipe.append(ttv2.RandomRotation(it['param']))
                case 'hflip':
                    # Horizontally flip the given image randomly with a given probability
                    pipe.append(ttv2.RandomHorizontalFlip(it['param']))
                case 'vflip':
                    # Vertically flip the given image randomly with a given probability
                    pipe.append(ttv2.RandomVerticalFlip(it['param']))
                case 'pad':
                    # Pad the given image on all sides with the given "pad" value
                    pipe.append(ttv2.Pad(it['param']))
                case 'brightness':
                    # Adjust brightness of the given image by factor
                    pipe.append(ttv2.ColorJitter(brightness=(it['param'])))
                case 'contrast':
                    # Adjust contrast of the given image by factor
                    pipe.append(ttv2.ColorJitter(contrast=(it['param'])))
                case 'saturation':
                    # Adjust color saturation of the given image by factor
                    pipe.append(ttv2.ColorJitter(saturation=(it['param'])))
                case 'hue':
                    # Adjust hue of the given image by factor
                    pipe.append(ttv2.ColorJitter(hue=(it['param'])))
                case 'grayscale':
                    # Convert image to grayscale
                    pipe.append(ttv2.Grayscale(it['param']))
                case 'blur':
                    # Blur the given image by a Gaussian filter of radius
                    pipe.append(ttv2.GaussianBlur(it['param']))
                case 'sharpness':
                    # Adjust the sharpness of the given image by factor
                    pipe.append(ttv2.RandomAdjustSharpness(it['param']))
                case 'normalize':
                    pipe.append(ttv2.Normalize(mean=eval(it['param']['mean']), std=eval(it['param']['std'])))

        # return transformer
        return ttv2.Compose(pipe)

    # transform data based on dataset field config
    def feature_eng(self, kind: str, params: dict, ds: ray.data.Dataset)->ray.data.Dataset:
        print(f'workflow>>>feature_eng: {ds.count()}')
        fe_ds: ray.data.Dataset = None
        match self.data_type:
            case 'DATA' | 'TIMESERIES':
                # tabular data transformation based on RAY Dataset
                fe_ds = ds
        print(f'workflow>>>>>>feature_eng: {fe_ds.count()}')
        return fe_ds


    # train ML algo based on ray and mlflow
    # dataset has fixed format for sklearn algo
    # {train_x:, train_y:, val_x:, val_y:}
    def ml_fit(self, kind: str, params: dict, ds: ray.data.Dataset):
        print(f'workflow>>>ml_fit: {ds.count()}')
        metrics_df = None
        df = ds.to_pandas()
        # shuffle data
        df = df.sample(frac=1).reset_index(drop=True)
        print(len(df))
        print(self.dataset_info.target)
        print(df)
        if kind == 'clf':
            s = ClassificationExperiment()
            s.setup(df, target=self.dataset_info.target[0], preprocess=False, normalize=False, train_size=0.8)
            best_model = s.compare_models()
            best_model = s.automl()
            # best_model = s.tune_model(best_model)
            # s.evaluate_model(best_model)
            final_model = s.finalize_model(best_model)
            prediction = s.predict_model(final_model)
            metrics_df = s.pull()
        elif kind == 'reg':
            s = RegressionExperiment()
            s.setup(df, target=self.dataset_info.target[0], preprocess=False, normalize=False, train_size=0.8)
            best_model = s.compare_models()
            best_model = s.automl()
            # best_model = s.tune_model(best_model)
            # s.evaluate_model(best_model)
            final_model = s.finalize_model(best_model)
            prediction = s.predict_model(final_model)
            metrics_df = s.pull()
        elif kind == 'cluster':
            s = ClusteringExperiment()
            s.setup(df, preprocess=False, normalize=False)
            md = s.create_model('kmeans')
            # s.evaluate_model(md)
            # s.plot_model(kmeans, plot = 'elbow')
            result = s.assign_model(md)
            metrics_df = s.pull()
        elif kind == 'anomaly':
            s = AnomalyExperiment()
            s.setup(df, preprocess=False, normalize=False)
            md = s.create_model('iforest')
            result = s.assign_model(md)
            metrics_df = s.pull()
        elif kind == 'ts_forecast':
            ts_date_field = [field['name'] for field in self.dataset_info.fields if
                             field['attr'] == 'date' and field.get('timeline')]
            if len(ts_date_field):
                df.set_index(ts_date_field[0], inplace=True)
            s = TSForecastingExperiment()
            s.setup(df, fh=6, fold=5, use_gpu=True, verbose=False)
            best_model = s.compare_models()
            final_model = s.finalize_model(best_model)
            prediction = s.predict_model(final_model, fh=24)
            metrics_df = s.pull()

        print(f'workflow>>>>>>ml_fit: done \n{metrics_df}')
        return metrics_df



