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
import s3fs
import duckdb


# This is very similar with data_loader.py
# It runs remotely on ray for async response and data load performance is lower than data_loader.py
# This is used for ML training. So the data load performance difference is not most important.
# Both this and data_loader.py should keep the same logic and bug fix
@ray.remote
class RayPipeline:
    def __init__(self, source_info: any, dataset_info: any = None, algo_info: any = None):
        self.source_info = source_info
        self.source_type = source_info.type.upper()
        self.dataset_info = dataset_info
        self.algo_info = algo_info
        self.transformed = False

        self.data_type = None
        if dataset_info:
            self.data_type = dataset_info.type.upper()

        # category = 'sklearn.classifier'
        # category = 'boost.xgboost'
        self.algo_cat = algo_info.category.upper()

        self.psw = None
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

    def load(self, content: str, params: []) -> ray.data.Dataset:
        # return ray dataset
        match self.source_type:
            case 'MYSQL':
                # return ray.data.Dataset
                return self.load_from_db(content, params)
            case 'S3BUCKET':
                # return ray.data.Dataset
                return self.load_from_bucket(content)
            case 'BUILDIN':
                # return ray.data.Dataset
                return self.load_from_buildin(content)


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
    def transform(self, ds: ray.data.Dataset):
        match self.data_type:
            case 'DATA':
                # tabular data transformation based on RAY Dataset
                return self.trans_data(ds)
            case 'IMAGE':
                # image data transformation based on RAY Dataset
                return self.trans_image(ds)



    # transform tabular data based on field config
    def trans_data(self, dataset: ray.data.Dataset)->dict:
        if dataset is None or dataset.count == 0:
            return None

        # get fields
        fields = self.dataset_info.fields
        omitFields = [field['name'] for field in fields if 'omit' in field]
        validFields = [field['name'] for field in fields if 'omit' not in field]
        targetFields = [field['name'] for field in fields if 'target' in field and 'omit' not in field]
        featureFields = list(set(validFields).difference(set(targetFields)))

        # remove omit fields
        ds = dataset.drop_columns(omitFields)

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

        ray_ds = None
        shuffle = self.algo_info.dataCfg.get('shuffle', False)
        ratio = self.algo_info.dataCfg.get('evalRatio', None)
        if ratio == 0:
            ratio = None

        if self.algo_cat.startswith('SKLEARN') or self.algo_cat.startswith('BOOST'):
            # shuffle and split data then build a dict of ray Dataset
            if targetFields:
                train_x, val_x, train_y, val_y = train_test_split(df[featureFields], df[targetFields],
                                                                  test_size=ratio, shuffle=shuffle)
                ray_ds = dict(train_x=ray.data.from_pandas(train_x), val_x=ray.data.from_pandas(val_x),
                              train_y=ray.data.from_pandas(train_y), val_y=ray.data.from_pandas(val_y))
            else:
                train_x, val_x = train_test_split(df[featureFields], test_size=ratio, shuffle=shuffle)
                ray_ds = dict(train_x=ray.data.from_pandas(train_x), val_x=ray.data.from_pandas(val_x))
        elif self.algo_cat.startswith('PYTORCH'):
            # pandas Dataframe -> torh tensor Dataset -> ray Dataset -> shuffle and split
            if targetFields:
                torch_ds = torch.utils.data.TensorDataset(torch.Tensor(df[featureFields].values),
                                                          torch.LongTensor(df[targetFields].to_numpy().ravel()))
            else:
                torch_ds = torch.utils.data.TensorDataset(torch.Tensor(df[featureFields].values))
            # Error converting data to Arrow: [(tensor([3.5000, 5.1000, 0.2000, 1.4000]), tensor(0))
            ds = ray.data.from_torch(torch_ds)
            train_ds, val_ds = ds.train_test_split(test_size=ratio, shuffle=shuffle)
            ray_ds = dict(train_set=train_ds, val_set=val_ds)

        # return a dict of ray Dataset
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
        ray_ds = dict(train_set=train_ds, val_set=val_ds)
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


    # train ML algo based on ray and mlflow
    # dataset has fixed format for sklearn algo
    # {train_x:, train_y:, val_x:, val_y:}
    def train(self, dataset: dict, train_func: any, params: dict):
        if self.algo_cat.startswith('SKLEARN'):
            return self.trainSklearn(params, train_func, dataset)
        elif self.algo_cat.startswith('PYTORCH'):
            if self.data_type == 'DATA':
                return self.trainTorchData(params, train_func, dataset)
            elif self.data_type == 'IMAGE':
                return self.trainTorchImage(params, train_func, dataset)
        elif self.algo_cat.endswith('XGBOOST'):
            return self.trainXGBoost(params, train_func, dataset)
        elif self.algo_cat.endswith('LIGHTGBM'):
            return self.trainXGBoost(params, train_func, dataset)


    # train sklearn algo for tabular data
    # MLflow sklearn autologging is known to be compatible with 0.24.1 <= scikit-learn <= 1.5.2
    def trainSklearn(self, params: dict, train_func, dataset: dict):
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

        # create progress report
        progressRpt = RayReport(params)
        progressRpt.jobProgress(JOB_PROGRESS_START)

        if params.get('gpu', False):
            tune_func = tune.with_resources(tune.with_parameters(train_func, data=dataset), resources={"gpu": 1})
        else:
            tune_func = tune.with_parameters(train_func, data=dataset)

        tune_cfg = tune.TuneConfig(num_samples=params['trials'],
                                   search_alg=search.BasicVariantGenerator(max_concurrent=1),
                                   scheduler=schedule.ASHAScheduler(mode="max"),
                                   time_budget_s=params['timeout'] * 60 * params['trials'] if params.get('timeout') else None)
        # ray will save tune results into storage_path with sub-folder exper_name
        # this is not used because we are using mlflow to save result on S3
        run_cfg = train.RunConfig(name=params['exper_name'],  stop=params.get('stop'),
                                  verbose=get_air_verbosity(AirVerbosity.DEFAULT),
                                  log_to_file=False, storage_path=TEMP_DIR+'/tune/',
                                  checkpoint_config=train.CheckpointConfig(checkpoint_frequency=0),
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
            progressRpt.jobProgress(JOB_PROGRESS_END, e)
        else:
            # report progress
            progressRpt.jobProgress(JOB_PROGRESS_END)



    # train ML algo based on ray and mlflow
    def trainPyTorch_bk(self, params: dict, train_func, data: any):
        # use AWS S3/minio as artifact repository
        os.environ["AWS_ACCESS_KEY_ID"] = params.get('s3_id')
        os.environ["AWS_SECRET_ACCESS_KEY"] = params.get('s3_key')
        os.environ["MLFLOW_S3_IGNORE_TLS"] = 'true'
        # os.environ["AWS_DEFAULT_REGION"] = None
        mlflow.environment_variables.MLFLOW_S3_IGNORE_TLS = 'true'
        mlflow.environment_variables.MLFLOW_S3_ENDPOINT_URL = params.get('s3_url')
        # use mysql db as tracking store
        mlflow.set_tracking_uri(params['tracking_url'])
        # resolve warning 'Experiment state snapshotting has been triggered multiple times in the last 5.0 seconds'
        os.environ['TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S'] = '0'
        # to disable log deduplication
        os.environ["RAY_DEDUP_LOGS"] = "0"
        # enable custom ProgressReporter when set to 0
        os.environ['RAY_AIR_NEW_OUTPUT'] = '0'
        # os.environ["WORLD_SIZE"] = '1'
        # os.environ["RANK"] = '0'

        params['tune_param']['dist'] = True

        # create a new experiment with UNIQUE name for mlflow (ex: algo_3_1234567890)
        exper_tags = dict(org_id=params['org_id'], algo_id=params['algo_id'], algo_name=params['algo_name'],
                              user_id=params['user_id'], args='|'.join(params['args']))
        params['exper_id'] = mlflow.create_experiment(name=params['exper_name'], tags=exper_tags,
                                                          artifact_location=params['artifact_location'])

        params['tune_param']['exper_id'] = params['exper_id']
        params['tune_param']['data'] = data
        # resolve the warning 'Matplotlib GUI outside of the main thread will likely fail'
        # matplotlib.use('agg')
        # mlflow.autolog()

        # create progress report
        progressRpt = RayReport(params)
        progressRpt.jobProgress(JOB_PROGRESS_START)

        # Configure computation resources
        scaling_cfg = ScalingConfig(num_workers=1, use_gpu=True)
        torch_cfg = TorchConfig(backend="gloo")
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            scaling_config=scaling_cfg,
            torch_config=torch_cfg
        )

        # ray will save tune results into storage_path with sub-folder exper_name
        # this is not used because we are using mlflow to save result on S3
        # earlystop will cause run.status is still running and end_time will be null
        tune_cfg = tune.TuneConfig(num_samples=params['trials'],
                                   search_alg=search.BasicVariantGenerator(max_concurrent=3))
        run_cfg = train.RunConfig(name=params['exper_name'],  # stop=params.get('stop'),
                                    checkpoint_config=train.CheckpointConfig(checkpoint_frequency=0),
                                    log_to_file=False, storage_path=TEMP_DIR + '/tune/',
                                    callbacks=[progressRpt])

        tuner = tune.Tuner(trainable=trainer,
                            tune_config=tune_cfg,
                           run_config=run_cfg,
                           param_space={"train_loop_config": params['tune_param']})
        try:
            # start train......
            result = tuner.fit()
        except RayError as e:
            print(e)
            progressRpt.jobProgress(JOB_PROGRESS_END, e)
        else:
            # report progress
            progressRpt.jobProgress(JOB_PROGRESS_END)


    # train pytorch algo for tabular data
    def trainTorchData(self, params: dict, train_func, dataset: dict):
        # use AWS S3/minio as artifact repository
        os.environ["AWS_ACCESS_KEY_ID"] = params.get('s3_id')
        os.environ["AWS_SECRET_ACCESS_KEY"] = params.get('s3_key')
        os.environ["MLFLOW_S3_IGNORE_TLS"] = 'true'
        # os.environ["AWS_DEFAULT_REGION"] = None
        mlflow.environment_variables.MLFLOW_S3_IGNORE_TLS = 'true'
        mlflow.environment_variables.MLFLOW_S3_ENDPOINT_URL = params.get('s3_url')
        # use mysql db as tracking store
        mlflow.set_tracking_uri(params['tracking_url'])
        # resolve warning 'Experiment state snapshotting has been triggered multiple times in the last 5.0 seconds'
        os.environ['TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S'] = '0'
        # to disable log deduplication
        os.environ["RAY_DEDUP_LOGS"] = "0"
        # enable custom ProgressReporter when set to 0
        os.environ['RAY_AIR_NEW_OUTPUT'] = '0'
        # os.environ["WORLD_SIZE"] = '1'
        # os.environ["RANK"] = '0'

        params['tune_param']['dist'] = True

        # create a new experiment with UNIQUE name for mlflow (ex: algo_3_1234567890)
        exper_tags = dict(org_id=params['org_id'], algo_id=params['algo_id'], algo_name=params['algo_name'],
                          user_id=params['user_id'], args='|'.join(params['args']))
        params['exper_id'] = mlflow.create_experiment(name=params['exper_name'], tags=exper_tags,
                                                      artifact_location=params['artifact_location'])

        params['tune_param']['exper_id'] = params['exper_id']

        # create progress report
        progressRpt = RayReport(params)
        progressRpt.jobProgress(JOB_PROGRESS_START)

        # Configure computation resources
        scaling_cfg = ScalingConfig(num_workers=1, use_gpu=params.get('gpu', False))
        torch_cfg = TorchConfig(backend="gloo")
        # ray will save tune results into storage_path with sub-folder exper_name
        # this is not used because we are using mlflow to save result on S3
        # earlystop will cause run.status is still running and end_time will be null
        tune_cfg = tune.TuneConfig(num_samples=params['trials'],
                                   search_alg=search.BasicVariantGenerator(max_concurrent=3))
        run_cfg = train.RunConfig(name=params['exper_name'],  # stop=params.get('stop'),
                                  checkpoint_config=train.CheckpointConfig(checkpoint_frequency=0),
                                  log_to_file=False, storage_path=TEMP_DIR + '/tune/',
                                  callbacks=[progressRpt])

        tuner = TorchTrainer(
            train_func,
            train_loop_config=params['tune_param'],
            scaling_config=scaling_cfg,
            run_config=run_cfg,
            torch_config=torch_cfg,
            datasets=dataset,
            dataset_config=ray.train.DataConfig(datasets_to_split=['train_set'])
        )


        try:
            # start train......
            result = tuner.fit()
        except RayError as e:
            print(e)
            progressRpt.jobProgress(JOB_PROGRESS_END, e)
        else:
            # report progress
            progressRpt.jobProgress(JOB_PROGRESS_END)


    # train pytorch algo for image data
    def trainTorchImage(self, params: dict, train_func, dataset: dict):
        # use AWS S3/minio as artifact repository
        os.environ["AWS_ACCESS_KEY_ID"] = params.get('s3_id')
        os.environ["AWS_SECRET_ACCESS_KEY"] = params.get('s3_key')
        os.environ["MLFLOW_S3_IGNORE_TLS"] = 'true'
        # os.environ["AWS_DEFAULT_REGION"] = None
        mlflow.environment_variables.MLFLOW_S3_IGNORE_TLS = 'true'
        mlflow.environment_variables.MLFLOW_S3_ENDPOINT_URL = params.get('s3_url')
        # use mysql db as tracking store
        mlflow.set_tracking_uri(params['tracking_url'])
        # resolve warning 'Experiment state snapshotting has been triggered multiple times in the last 5.0 seconds'
        os.environ['TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S'] = '0'
        # to disable log deduplication
        os.environ["RAY_DEDUP_LOGS"] = "0"
        # enable custom ProgressReporter when set to 0
        os.environ['RAY_AIR_NEW_OUTPUT'] = '0'
        # os.environ["WORLD_SIZE"] = '1'
        # os.environ["RANK"] = '0'

        # create a new experiment with UNIQUE name for mlflow (ex: algo_3_1234567890)
        exper_tags = dict(org_id=params['org_id'], algo_id=params['algo_id'], algo_name=params['algo_name'],
                          user_id=params['user_id'], args='|'.join(params['args']))
        params['exper_id'] = mlflow.create_experiment(name=params['exper_name'], tags=exper_tags,
                                                      artifact_location=params['artifact_location'])

        params['tune_param']['exper_id'] = params['exper_id']
        params['tune_param']['dist'] = True

        # create progress report
        progressRpt = RayReport(params)
        progressRpt.jobProgress(JOB_PROGRESS_START)

        if params['metrics'] and params['threshold']:
            # stop when the metric mets the threshold
            if params['metrics'] in ['accuracy', 'f1']:
                # The bigger the better. mode is fixed 'max' in RunConfig.stop
                early_stopper = {params['metrics']: params['threshold'], "training_iteration": params['epochs']}
                if params['timeout']:
                    early_stopper['time_total_s'] = params['timeout'] * 60
            else:
                # The smaller the better. define a custom TrialPlateauStopper with mode 'min'
                early_stopper = TrialPlateauStopper(metric=params['metrics'], mode="min",
                                                    metric_threshold=params['threshold'])
        else:
            early_stopper = {"training_iteration": params['epochs']}
            if params['timeout']:
                early_stopper['time_total_s'] = params['timeout'] * 60

        # Configure computation resources
        scaling_cfg = ScalingConfig(num_workers=1, use_gpu=params['gpu'])
        torch_cfg = TorchConfig(backend="gloo")
        # ray will save tune results into storage_path with sub-folder exper_name
        # this is not used because we are using mlflow to save result on S3
        # earlystop will cause run.status is still running and end_time will be null
        run_cfg = train.RunConfig(name=params['exper_name'],
                                  stop=early_stopper,
                                  checkpoint_config=train.CheckpointConfig(checkpoint_frequency=0),
                                  failure_config=train.FailureConfig(fail_fast=True),
                                  log_to_file=False,
                                  storage_path=TEMP_DIR + '/tune/',
                                  callbacks=[progressRpt])

        tuner = TorchTrainer(
            train_func,
            train_loop_config=params['tune_param'],
            scaling_config=scaling_cfg,
            run_config=run_cfg,
            torch_config=torch_cfg,
            datasets=dataset,
            dataset_config=ray.train.DataConfig(datasets_to_split=['train_set'])
        )

        try:
            # start train......
            result = tuner.fit()
        except RayError as e:
            print(e)
            progressRpt.jobProgress(JOB_PROGRESS_END, e)
        else:
            # report progress
            progressRpt.jobProgress(JOB_PROGRESS_END)


    # train ML algo based on ray and mlflow
    def trainXGBoost(self, params: dict, train_func, dataset: any):
        # use AWS S3/minio as artifact repository
        os.environ["AWS_ACCESS_KEY_ID"] = params.get('s3_id')
        os.environ["AWS_SECRET_ACCESS_KEY"] = params.get('s3_key')
        os.environ["MLFLOW_S3_IGNORE_TLS"] = 'true'
        # os.environ["AWS_DEFAULT_REGION"] = None
        mlflow.environment_variables.MLFLOW_S3_IGNORE_TLS = 'true'
        mlflow.environment_variables.MLFLOW_S3_ENDPOINT_URL = params.get('s3_url')
        # use mysql db as tracking store
        mlflow.set_tracking_uri(params['tracking_url'])
        # resolve warning 'Experiment state snapshotting has been triggered multiple times in the last 5.0 seconds'
        os.environ['TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S'] = '0'
        # to disable log deduplication
        os.environ["RAY_DEDUP_LOGS"] = "0"
        # enable custom ProgressReporter when set to 0
        os.environ['RAY_AIR_NEW_OUTPUT'] = '0'
        # os.environ["WORLD_SIZE"] = '1'
        # os.environ["RANK"] = '0'

        params['tune_param']['dist'] = True

        # create a new experiment with UNIQUE name for mlflow (ex: algo_3_1234567890)
        exper_tags = dict(org_id=params['org_id'], algo_id=params['algo_id'], algo_name=params['algo_name'],
                          user_id=params['user_id'], args='|'.join(params['args']))
        params['exper_id'] = mlflow.create_experiment(name=params['exper_name'], tags=exper_tags,
                                                      artifact_location=params['artifact_location'])
        params['tune_param']['exper_id'] = params['exper_id']

        # create progress report
        progressRpt = RayReport(params)
        progressRpt.jobProgress(JOB_PROGRESS_START)

        if params['gpu'] and RAY_NUM_GPU > 0:
            # tune function with resources
            tune_func = tune.with_resources(tune.with_parameters(train_func, data=dataset), resources={"gpu": RAY_NUM_GPU})
        else:
            tune_func = tune.with_parameters(train_func, data=dataset)

        scheduler_cfg = None
        if params['metrics'] and params['threshold']:
            # stop when the metric mets the threshold
            if params['metrics'] in ['accuracy', 'f1']:
                # The bigger the better. mode is fixed 'max' in RunConfig.stop
                early_stopper = {f"validation_0-{params['metrics']}" : params['threshold'], "training_iteration": params['epochs']}
                if params['timeout']:
                    early_stopper['time_total_s'] = params['timeout'] * 60
            else:
                # The smaller the better. define a custom TrialPlateauStopper with mode 'min'
                early_stopper = TrialPlateauStopper(metric=f"validation_0-{params['metrics']}", mode="min",
                                                    metric_threshold=params['threshold'])
                # metric: is mandatory. like 'validation_0-rmse' for xgboost
                # the main purpose is to stop the trial when max_t reaches the max epochs
                scheduler_cfg = schedule.ASHAScheduler(max_t=params['epochs'], grace_period=3,
                                                       metric=f"validation_0-{params['metrics']}", mode='min')
        else:
            early_stopper = {"training_iteration": params['epochs']}
            if params['timeout']:
                early_stopper['time_total_s'] = params['timeout'] * 60


        # time_budget_s: stop training when time budget (in seconds) has elapsed for a trail
        # max_t: stop training when epoch reaches max_t
        tune_cfg = tune.TuneConfig(num_samples=params['trials'],
                                   search_alg=search.BasicVariantGenerator(max_concurrent=3),
                                   scheduler=scheduler_cfg,
                                   time_budget_s=params['timeout'] * 60 if params['timeout'] else None)
        # ray will save tune results into storage_path with sub-folder exper_name
        # this is not used because we are using mlflow to save result on S3
        run_cfg = train.RunConfig(name=params['exper_name'],
                                  stop=early_stopper,
                                  verbose=get_air_verbosity(AirVerbosity.DEFAULT),
                                  log_to_file=False, storage_path=TEMP_DIR + '/tune/',
                                  failure_config=train.FailureConfig(fail_fast=True),
                                  checkpoint_config=train.CheckpointConfig(checkpoint_frequency=0),
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
            progressRpt.jobProgress(JOB_PROGRESS_END, e)
        else:
            # report progress
            progressRpt.jobProgress(JOB_PROGRESS_END)


    # train ML algo based on ray and mlflow
    def run(self, train_func: any, params: dict):
        ray_ds:ray.data.Dataset = self.load(self.dataset_info.content, self.dataset_info.variable)
        transformed_ds:dict = self.transform(ray_ds)
        self.train(transformed_ds, train_func, params)


