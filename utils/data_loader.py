#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/9/13
# @desc           : load data from DB or S3

import base64
import boto3
import mysql.connector as ct
import pandas as pd
import duckdb
import io
import ray
from sqlalchemy.ext.asyncio import create_async_engine
from config.settings import TEMP_DIR
from core.logger import logger
import torch
import torchvision
import torchaudio
from sklearn import preprocessing as pp
from sklearn import feature_extraction as fe
from pandas import CategoricalDtype
from sqlalchemy import text

# This is very similar with ray_pipeline.py
# It runs locally for synchronized response(await) and performance is higher than ray_pipeline.py
# As reference some different methods are used here
# Both this and ray_pipeline should keep the same logic and bug fix
class DataLoader:
    def __init__(self, source_info: any, dataset_info: any = None):
        self.source_info = source_info
        self.source_type = source_info.type.upper()
        self.dataset_info = dataset_info

        self.data_type = None
        if dataset_info:
            self.data_type = dataset_info.type.upper()

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

                # create db engine
                self.engine = create_async_engine(
                    f'mysql+asyncmy://{source_info.username}:{self.psw}@{source_info.url}?charset=utf8mb4', echo=False)
            case 'S3BUCKET':
                # get bucket info (endpoint and bucket name)
                https = source_info.url.split('//')
                urls = https[1].split('/')
                self.bucket = urls[1]

                # create S3 engine
                self.s3fs = boto3.client('s3', endpoint_url=f'{https[0]}//{urls[0]}', aws_access_key_id=source_info.username, aws_secret_access_key=self.psw)
            case 'BUILDIN':
                # for pytorch, huggingface, etc.
                nothing = None

    """
    load data from source and return pandas dataframe
    """
    async def load(self, content: str, params: []) -> pd.DataFrame:
        try:
            match self.source_type:
                case 'MYSQL':
                    return await self.load_from_db(content, params)
                case 'S3BUCKET':
                    return await self.load_from_bucket(content)
                case 'BUILDIN':
                    return await self.load_from_buildin(content)
                case '_':
                    return None
        except Exception as e:
            logger.error(f"Failed to load data，{e.__str__()}")
            raise ValueError(f"Failed to load data！{e.__str__()}")

    async def load_from_db(self, content: str, params: []) -> pd.DataFrame:
        # df = pd.read_sql(content, self.engine)
        async with self.engine.connect() as conn:
            result = await conn.execute(text(content))
        col_names = result.keys()
        data = result.fetchall()
        df = pd.DataFrame.from_records(data, columns=col_names)
        return df

    async def load_from_db_2(self, content: str, params: []) -> pd.DataFrame:
        # serialization issue is resolved by this way
        def create_conn(usr: str, psw: str, host: str, db: str):
            def conn():
                return ct.connect(user=usr, password=psw, host=host, database=db)
            return conn

        conn_factory = create_conn(self.source_info.username, self.psw, self.host, self.db)
        ds = ray.data.read_sql(content, conn_factory, override_num_blocks=1)
        return ds.to_pandas()

    async def load_from_bucket(self, content: str)->pd.DataFrame:
        # find files from content and get file list
        # ex: SELECT * FROM 'mldata/iris.csv'
        file_list = [f.strip("'") for f in content.split(' ') if
                     f.upper().endswith((".CSV'", ".JSON'", ".PARQUET'", ".TXT'"))]
        # get unique files
        file_list = list(set(file_list))
        duck_sql = content
        dfn = {}
        df0 = df1 = df2 = None
        for idx, f_name in enumerate(file_list):
            obj = self.s3fs.get_object(Bucket=self.bucket, Key=f_name)
            if f_name.upper().endswith(('.CSV')):
                match idx:
                    case 0:
                        df0 = pd.read_csv(obj['Body'])
                    case 1:
                        df1 = pd.read_csv(obj['Body'])
                    case 2:
                        df2 = pd.read_csv(obj['Body'])
            elif f_name.upper().endswith(('.JSON')):
                match idx:
                    case 0:
                        df0 = pd.read_json(io.StringIO(obj['Body'].read().decode('utf-8')))
                    case 1:
                        df1 = pd.read_json(io.StringIO(obj['Body'].read().decode('utf-8')))
                    case 2:
                        df2 = pd.read_json(io.StringIO(obj['Body'].read().decode('utf-8')))
            duck_sql = duck_sql.replace(f_name, f'df{idx}')
        return duckdb.sql(duck_sql).df()


    async def load_from_bucket_2(self, content: str)->pd.DataFrame:
        # find files from content and get file list
        # ex: SELECT * FROM 'mldata/iris.csv'
        file_list = [f.strip("'") for f in content.split(' ') if
                     f.upper().endswith((".CSV'", ".JSON'", ".PARQUET'", ".TXT'"))]
        # get unique files
        file_list = list(set(file_list))
        duck_sql = content
        dfn = {}
        df0 = df1 = df2 = None

        for idx, f_name in enumerate(file_list):
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


    async def load_from_buildin(self, content: str):
        if content.startswith('sklearn'):
            import sklearn
            from sklearn import datasets
            dataset = eval(content)()
            df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            if 'target' in dataset.keys():
                df['target'] = dataset.target
            return df
        elif content.startswith('torchvision'):
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.5,), (0.5,))])
            # it will be downloaded if it doesn't exist in local folder
            data = eval(content)(TEMP_DIR + '/data/', download=True, transform=None)
            return data
        elif content.startswith('torchaudio'):
            transform = torchaudio.transforms.Compose([torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize((0.5,), (0.5,))])
            # it will be downloaded if it doesn't exist in local folder
            data = eval(content)(TEMP_DIR + '/data/', download=True, transform=None)
            return data

    """
    transform data
    """
    async def transform(self, df: pd.DataFrame, fields: any):
        match self.data_type:
            case 'DATA':
                return self.transTabular(df, fields)
            case 'IMAGE':
                return self.transTorch(df, fields)
            case 'AUDIO':
                return self.transTorch(df, fields)
            case 'VIDEO':
                return self.transTorch(df, fields)

    # transform tabular data based on field config
    def transTabular(self, df: pd.DataFrame, fields):
        if df is None or len(df) == 0:
            return None

        # get fields
        omitFields = [field['name'] for field in fields if 'omit' in field]
        validFields = [field['name'] for field in fields if 'omit' not in field]
        targetFields = [field['name'] for field in fields if 'target' in field and 'omit' not in field]
        featureFields = list(set(validFields).difference(set(targetFields)))

        # remove omit fields
        df.drop(columns=omitFields, inplace=True)

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

        return df





    """
    transform data
    """

    async def transform_bk(self, df: pd.DataFrame, fields: any):
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

        targets = [field['name'] for field in fields if 'target' in field and 'omit' not in field]
        # Split the data into train and test sets.
        cols = df.columns.tolist()
        features = list(set(cols).difference(set(targets)))
        # x, tx, y, ty = train_test_split(df[features], df[targets], test_size=ratio, shuffle=shuffle)
        # data: dict = {'x': x, 'y': y, 'tx': tx, 'ty': ty}

        tensor_data = torch.utils.data.TensorDataset(torch.Tensor(df[features].values), torch.LongTensor(df[targets].to_numpy().ravel()))
        # train_set, val_set = torch.utils.data.random_split(df, [1 - ratio, ratio])
        # self.trainset = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=shuffle)
        # self.evalset = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=shuffle)
        # data = {'train': self.trainset, 'eval': self.evalset}

        return tensor_data

    async def run(self):
        df = await self.load(self.dataset_info.content, self.dataset_info.variable)
        return await self.transform(df, self.dataset_info.fields)