#!/usr/bin/python
# -*- coding: utf-8 -*-
# @version        : 1.0
# @Create Time    : 2024/9/15
# @desc           : ray data loader

import base64
import boto3
import mysql.connector
import pandas as pd
import duckdb
import io
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from config.settings import TEMP_DIR
from core.logger import logger
from sqlalchemy import text
import sklearn
import torch
import torchvision.datasets
import torchaudio
import ray
from pandas import CategoricalDtype
from sklearn import preprocessing as pp
from sklearn import feature_extraction as fe
from sklearn.model_selection import train_test_split



@ray.remote
class RayDataLoader:
    def __init__(self, source: any):
        self.type = source.type.upper()
        self.source = source
        self.psw = None
        if source.password and source.password.upper() != 'NA' and source.password != 'NONE':
            self.psw = base64.b64decode(source.password).decode('utf-8')

        match self.type:
            case 'MYSQL':
                '''
                self.conn = mysql.connector.connect(
                    user=source.username,
                    password=self.psw,
                    host='datapie.cjiaoci4g12w.us-east-1.rds.amazonaws.com',
                    database='datapie',
                    connection_timeout=30
                )
                '''
                self.engine = create_engine(
                    f'mysql+mysqldb://{source.username}:{self.psw}@{source.url}?charset=utf8mb4', echo=False)
            case 'S3BUCKET':
                https = source.url.split('//')
                urls = https[1].split('/')
                self.bucket = urls[1]
                self.s3fs = boto3.client('s3', endpoint_url=f'{https[0]}//{urls[0]}', aws_access_key_id=source.username, aws_secret_access_key=self.psw)
            case 'BUILDIN':
                # do nothing
                nothing = None


    def load(self, content: str, params: any, limit: int = None):
        df = None
        match self.type:
            case 'MYSQL':
                return pd.read_sql(content, self.engine)
            case 'S3BUCKET':
                file_list = [f for f in content.split(' ') if f.endswith(('.csv', '.CSV', '.json', '.JSON'))]
                duck_sql = content
                dfn = {}
                df0 = df1 = df2 = df3 = None
                for idx, f_name in enumerate(file_list):
                    obj = self.s3fs.get_object(Bucket=self.bucket, Key=f_name)
                    if f_name.endswith(('.csv', '.CSV')):
                        match idx:
                            case 0:
                                df0 = pd.read_csv(obj['Body'])
                            case 1:
                                df1 = pd.read_csv(obj['Body'])
                            case 2:
                                df2 = pd.read_csv(obj['Body'])
                            case 3:
                                df3 = pd.read_csv(obj['Body'])
                    elif f_name.endswith(('.json', '.JSON')):
                        match idx:
                            case 0:
                                df0 = pd.read_json(io.StringIO(obj['Body'].read().decode('utf-8')))
                            case 1:
                                df1 = pd.read_json(io.StringIO(obj['Body'].read().decode('utf-8')))
                            case 2:
                                df2 = pd.read_json(io.StringIO(obj['Body'].read().decode('utf-8')))
                            case 3:
                                df3 = pd.read_json(io.StringIO(obj['Body'].read().decode('utf-8')))
                    duck_sql = duck_sql.replace(f_name, f'df{idx}')
                return duckdb.sql(duck_sql).df()
            case 'BUILDIN':
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
                    return eval(content)(TEMP_DIR + '/data/', download=True, transform=transform)
                elif content.startswith('torchaudio'):
                    transform = torchaudio.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize((0.5,), (0.5,))])
                    # it will be downloaded if it doesn't exist in local folder
                    return eval(content)(TEMP_DIR + '/data/', download=True, transform=transform)



    def db_close(self):
        self.connection.close()
        self.engine.dispose()


    # transform data based on dataset field config
    def transform(self, framework: str, df: pd.DataFrame, targets: list, fields, ratio: int, shuffle: bool):
        match framework:
            case 'sklearn':
                return self.transSk(df, fields, ratio, shuffle)
            case 'pytorch':
                return self.transTorch(df, targets, ratio, 64, shuffle)


    # transform/split/shuffle
    def transTorch(self, dataset, targets: list, ratio: int = 0.3, batch_size: int = 64, shuffle: bool = False):
        # Split the data into train and validation sets.
        train_set, val_set = torch.utils.data.random_split(dataset, [1-ratio, ratio])
        self.trainset = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        self.evalset = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)
        return {'train': self.trainset, 'eval': self.evalset}


    # transform data based on dataset field config
    def transSk(self, df: pd.DataFrame, fields, ratio: int, shuffle: bool):
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
        x, tx, y, ty = train_test_split(df[features], df[targets], test_size=ratio, shuffle=shuffle)
        data: dict = {'x': x, 'y': y.to_numpy().ravel(), 'tx': tx, 'ty': ty.to_numpy().ravel()}
        self.train_data = data
        return data