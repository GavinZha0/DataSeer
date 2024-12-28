#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/9/15
# @desc           : ray data loader

import base64
import boto3
import mysql.connector as ct
import pandas as pd
import duckdb
import io

import s3fs
from sqlalchemy import create_engine

from config import settings
from config.settings import TEMP_DIR
import torch
import torchvision.datasets
import torchaudio
import ray
from pandas import CategoricalDtype
from sklearn import preprocessing as pp
from sklearn import feature_extraction as fe
from sklearn.model_selection import train_test_split

# This is very similar with data_loader.py
# It runs remotely on ray for async response and data load performance is lower than data_loader.py
# This is used for ML training. So the data load performance difference is not most important.
# Both this and data_loader.py should keep the same logic and bug fix
@ray.remote
class RayDataLoader:
    def __init__(self, source: any, data_type: str, algo_frame: str):
        self.source_type = source.type.upper()
        self.data_type = data_type.upper()
        self.algo_frame = algo_frame.upper()
        self.source = source
        self.psw = None

        if source.password and source.password.upper() != 'NA' and source.password != 'NONE':
            # decode password
            self.psw = base64.b64decode(source.password).decode('utf-8')

        match self.source_type:
            case 'MYSQL':
                # get db info
                self.host = source.url.split(':')[0]
                tmp = source.url.split(':')[1]
                self.port = int(tmp.split('/')[0])
                self.db = tmp.split('/')[1]

                # self.engine = create_engine(f'mysql+mysqldb://{source.username}:{self.psw}@{source.url}?charset=utf8mb4', echo=False)
            case 'S3BUCKET':
                # get bucket info
                https = source.url.split('//')
                urls = https[1].split('/')
                self.bucket = urls[1]

                self.fs = s3fs.S3FileSystem(use_ssl=False, client_kwargs={
                        "aws_access_key_id": settings.AWS_S3_ACCESS_ID,
                        "aws_secret_access_key": settings.AWS_S3_SECRET_KEY,
                        "endpoint_url": settings.AWS_S3_ENDPOINT,
                        "verify": False}
                    )
                # self.s3fs = boto3.client('s3', endpoint_url=f'{https[0]}//{urls[0]}', aws_access_key_id=source.username, aws_secret_access_key=self.psw)
            case 'BUILDIN':
                # do nothing
                nothing = None


    def load(self, content: str, params: any, limit: int = None):
        df = None
        match self.source_type:
            case 'MYSQL':
                return self.load_from_db(content)
                # return pd.read_sql(content, self.engine)
            case 'S3BUCKET':
                # old method
                # obj = self.s3fs.get_object(Bucket=self.bucket, Key=f_name)
                # df0 = pd.read_csv(obj['Body'])
                # df0 = pd.read_json(io.StringIO(obj['Body'].read().decode('utf-8')))
                return self.load_from_bucket(content)
            case 'BUILDIN':
                return self.load_from_buildin(content)


    def load_from_db(self, content: str):
        # serialization issue of ray.data.read_sql is resolved by this way
        def create_conn(usr: str, psw: str, host: str, db: str):
            def conn():
                return ct.connect(user=usr, password=psw, host=host, database=db)
            return conn

        conn_factory = create_conn(self.source.username, self.psw, self.host, self.db)
        ds = ray.data.read_sql(content, conn_factory, override_num_blocks=1)
        # return ray dataset
        return ds


    def load_from_bucket(self, content: str):
        if self.data_type == 'IMAGE':
            # find folders from content and get folder list
            # ex: SELECT * FROM 'mldata/images/'
            folder_list = [f.strip("'") for f in content.split(' ') if f.lower().endswith(("/'"))]
            # get unique files
            folder_list = list(set(folder_list))
            if folder_list:
                f_name = folder_list[0]
                ds = ray.data.read_images(filesystem=self.fs, paths=f"s3://{self.bucket}/{f_name}")
                print(ds.schema())
                return ds
            else:
                return None
        elif self.data_type == 'DATA':
            # find files from content and get file list
            # ex: SELECT * FROM 'mldata/iris.csv'
            file_list = [f.strip("'") for f in content.split(' ') if f.lower().endswith((".csv'", ".json'", ".parquet'", ".txt'"))]
            # get unique files
            file_list = list(set(file_list))
            duck_sql = content
            dfn = {}
            df0 = df1 = df2 = None

            for idx, f_name in enumerate(file_list):
                if f_name.lower().endswith(('.csv')):
                    # get data from csv file using ray
                    ds = ray.data.read_csv(filesystem=self.fs, paths=f"s3://{self.bucket}/{f_name}")
                elif f_name.lower().endswith(('.json')):
                    # get data from json file using ray
                    ds = ray.data.read_json(filesystem=self.fs, paths=f"s3://{self.bucket}/{f_name}")
                elif f_name.lower().endswith(('.parquet')):
                    # get data from parquet file using ray
                    ds = ray.data.read_parquet(filesystem=self.fs, paths=f"s3://{self.bucket}/{f_name}")
                elif f_name.lower().endswith(('.txt')):
                    # get data from text file using ray
                    ds = ray.data.read_text(filesystem=self.fs, paths=f"s3://{self.bucket}/{f_name}")
                else:
                    continue

                # df = ds.to_pandas()
                # support 3 files only in content (usually there is 1 file)
                match idx:
                    case 0:
                        df0 = ds.to_pandas()
                    case 1:
                        df1 = ds.to_pandas()
                    case 2:
                        df2 = ds.to_pandas()
                    case '_':
                        continue

                # replace file name with dfX
                duck_sql = duck_sql.replace(f_name, f'df{idx}')
            # duckDB supports to query data from files using sql syntax
            # run sql to get data then convert to dataframe
            return duckdb.sql(duck_sql).df()

    def load_from_buildin(self, content: str):
        if content.startswith('sklearn'):
            dataset = eval(content)()
            df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            if 'target' in dataset.keys():
                df['target'] = dataset.target
            total = len(df)
        elif content.startswith('torchvision'):
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.5,), (0.5,))])
            # it will be downloaded if it doesn't exist in local folder
            data = eval(content)(TEMP_DIR + '/data/', download=True, transform=transform)
            ds = ray.data.from_torch(data)
            # print(ds)
            return ds
        elif content.startswith('torchaudio'):
            transform = torchaudio.transforms.Compose([torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize((0.5,), (0.5,))])
            # it will be downloaded if it doesn't exist in local folder
            data = eval(content)(TEMP_DIR + '/data/', download=True, transform=transform)
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
    def transform(self, ds: ray.data.dataset, fields: any):
        match self.data_type:
            case 'DATA':
                return self.transTabularData(ds, fields)
            case 'IMAGE':
                return self.transTorch(ds, fields)
            case 'AUDIO':
                return self.transTorch(ds, fields)
            case 'VIDEO':
                return self.transTorch(ds, fields)


    # transform/split/shuffle
    def transTorch(self, df: pd.DataFrame, fields):
        # Split the data into train and validation sets.
        # train_set, val_set = torch.utils.data.random_split(dataset, [1-ratio, ratio])
        # self.trainset = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        # self.evalset = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)
        # return {'train': self.trainset, 'eval': self.evalset}

        for it in fields:
            if it.get('omit'):
                # delete omit fields
                df.drop(columns=[it['name']], inplace=True)

        targets = [field['name'] for field in fields if 'target' in field and 'omit' not in field]
        # Split the data into train and test sets.
        cols = df.columns.tolist()
        features = list(set(cols).difference(set(targets)))
        print(features)
        print(targets)
        tensor_data = torch.utils.data.TensorDataset(torch.Tensor(df[features].values),
                                                     torch.LongTensor(df[targets].to_numpy().ravel()))
        return tensor_data


    # transform data based on dataset field config
    def transSk(self, ds: any, fields, ratio: int, shuffle: bool):
        # convert ray dataset to pandas dataframe
        # pandas has more methods to transform data
        # Polars is a replacement of pandas for data analysis. it is faster than pandas.
        df: pd.DataFrame = ds.to_pandas()

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
                    df[field_name] = pp.KBinsDiscretizer(n_bins=10, strategy='uniform',
                                                         encode='ordinal').fit_transform(
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
        tensor_data = torch.utils.data.TensorDataset(torch.Tensor(df[features].values), torch.LongTensor(df[targets].to_numpy().ravel()))
        return tensor_data

    # transform tabular data based on field config
    def transTabularData(self, dataset: ray.data.Dataset, fields):
        if dataset is None or dataset.count == 0:
            return None

        # get fields
        omitFields = [field['name'] for field in fields if 'omit' in field]
        validFields = [field['name'] for field in fields if 'omit' not in field]
        targetFields = [field['name'] for field in fields if 'target' in field and 'omit' not in field]
        featureFields = list(set(validFields).difference(set(targetFields)))

        # remove omit fields
        dataset = dataset.drop_columns(omitFields)

        # convert ray dataset to pandas dataframe
        # pandas has more methods to transform data
        # Polars is a replacement of pandas for data analysis. it is faster than pandas.
        df: pd.DataFrame = dataset.to_pandas()

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
                        df[field_name] = pp.OrdinalEncoder().fit_transform(df[field_name])
                case 'hot':  # One-Hot
                    df[field_name] = pp.OneHotEncoder().fit_transform(df[field_name])
                case 'hash':  # Hashing
                    df[field_name] = fe.FeatureHasher(input_type='string').fit_transform(df[field_name])
                case 'binary':  # Binary
                    df[field_name] = pp.Binarizer(threshold=1).fit_transform(df[field_name])
                case 'bins':  # Binning
                    df[field_name] = pp.KBinsDiscretizer(n_bins=10, strategy='uniform',
                                                         encode='ordinal').fit_transform(
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

        # convert dataset to match target framework
        final_ds = None
        match self.algo_frame:
            case 'SKLEARN':
                # convert to ray dataset
                final_ds = ray.data.from_pandas(df)
            case 'PYTORCH':
                # convert to tensor dataset
                if targetFields:
                    final_ds = torch.utils.data.TensorDataset(torch.Tensor(df[featureFields].values),
                                            torch.LongTensor(df[targetFields].to_numpy().ravel()))
                else:
                    final_ds = torch.utils.data.TensorDataset(torch.Tensor(df[featureFields].values))
            case '_':
                return None
        return final_ds