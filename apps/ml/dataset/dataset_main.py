import io
import base64
import dateparser
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
import torchaudio
from config.settings import TEMP_DIR
from core.logger import error


async def buildin_dataset_load(func: str, limit: int, params: []):
    if func.startswith('sklearn'):
        dataset = eval(func)()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        if 'target' in dataset.keys():
            df['target'] = dataset.target
        return df, len(df)
    elif func.startswith('torchvision'):
        # it will be downloaded if it doesn't exist in local folder
        dataset = eval(func)(TEMP_DIR + '/data/', download=True)
        return dataset, len(dataset)
    elif func.startswith('torchaudio'):
        # it will be downloaded if it doesn't exist in local folder
        dataset = eval(func)(TEMP_DIR + '/data/', download=True)
        return dataset, len(dataset)




async def get_data_stat(type: str, df: any, limit: int = None):
    data = None
    stat = None
    max_limit = limit
    if limit == None:
        # default limit
        max_limit = 10

    match(type):
        case 'data' | 'timeseries':
            # detect datetime fields
            type_list = []
            from datetime import date, datetime, timedelta, time
            for col in df.columns:
                if df[col].dtypes.name in ['object']:
                    obj_cell = df[col].dropna().iloc[0]
                    if pd.api.types.is_datetime64_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_datetime64_ns_dtype(df[col]):
                        type_list.append('datetime')
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif isinstance(obj_cell, date):
                        type_list.append('date')
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif isinstance(obj_cell, time):
                        type_list.append('time')
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif isinstance(obj_cell, datetime) or isinstance(obj_cell, timedelta):
                        type_list.append('datetime')
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif len(obj_cell) > 4 and isinstance(obj_cell, str):
                        digits = [1 if c.isdigit() else 0 for c in obj_cell]
                        digit_count = sum(digits)
                        if digit_count >= 4:
                            # try to parse datetime string
                            date_time = dateparser.parse(obj_cell)
                            if date_time is not None:
                                type_list.append('datetime')
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                        else:
                            type_list.append(df[col].dtypes.name)
                    else:
                        type_list.append(df[col].dtypes.name)
                else:
                    type_list.append('datetime' if df[col].dtypes.name.startswith('datetime') else df[col].dtypes.name)

            # get stat info
            desc = df.describe(include='all')
            # convert datetime(like Timestamp('2024-11-15 06:00:00')) to string for dict to json
            # otherwise, it will fail (TypeError: Type is not JSON serializable: Timestamp)
            # date_types = ['datetime' if it.name.startswith('datetime') else it.name for it in df.dtypes]
            ts_col = df.select_dtypes(include='datetime').columns.tolist()
            if ts_col is not None:
                for col in ts_col:
                    df[col] = df[col].astype('str')
                    desc[col] = desc[col].replace(np.nan, '')
                    desc[col] = desc[col].astype('str')

                    # remove ms (.SSS)
                    if '.' in desc[col]['min']:
                        desc.loc['min', col] = desc[col]['min'][:desc[col]['min'].index('.')]
                    if '.' in desc[col]['25%']:
                        desc.loc['25%', col] = desc[col]['25%'][:desc[col]['25%'].index('.')]
                    if '.' in desc[col]['mean']:
                        desc.loc['mean', col] = desc[col]['mean'][:desc[col]['mean'].index('.')]
                    if '.' in desc[col]['50%']:
                        desc.loc['50%', col] = desc[col]['50%'][:desc[col]['50%'].index('.')]
                    if '.' in desc[col]['75%']:
                        desc.loc['75%', col] = desc[col]['75%'][:desc[col]['75%'].index('.')]
                    if '.' in desc[col]['max']:
                        desc.loc['max', col] = desc[col]['max'][:desc[col]['max'].index('.')]

            # round values
            stat = desc.round(3).T
            stat['type'] = type_list
            stat['missing'] = df.isnull().sum().values
            # get unique values for each column
            stat['unique'] = {}
            stat['nunique'] = {}
            for col in df.columns:
                stat.loc[col, 'nunique'] = df[col].nunique()
                if stat.loc[col, 'nunique'] <= 20:
                    # list the unique values
                    if df[col].dtypes.name in ['int64', 'float64']:
                        stat.at[col, 'unique'] = np.sort(df[col].dropna().unique()).tolist()
                    elif df[col].dtypes.name in ['object']:
                        stat.at[col, 'unique'] = np.sort(df[col].dropna().unique().astype('str').tolist()).tolist()
                else:
                    stat.at[col, 'unique'] = []
            stat_var = df.var(numeric_only=True).to_frame('variance')
            stat = pd.merge(stat, stat_var, left_index=True, right_index=True, how='outer')
            stat = stat.reset_index().rename(columns={"index": "name", "25%": "pct25", "50%": "median", "75%": "pct75"})
            stat = stat.round(3)
            # convert to dict
            # stat.fillna(0, inplace=True)
            stat = stat.to_dict(orient='records')
            # preview data
            data = df.head(max_limit)
        case 'image':
            # get image attributes as stat info
            img, label = df[0]
            stat_img = dict(name='image', type='int', count=len(df), mode=img.mode, channel=len(img.split()), size=img.size)
            # uniques = df.classes
            uniques = df.targets.unique().tolist()
            uniques.sort()
            stat_label = dict(name='label', type='int', count=len(df), nunique=len(uniques), unique=uniques)
            stat = [stat_img, stat_label]

            # encode image to base64 string
            data = pd.DataFrame(data=None, columns=['image', 'label'])
            for i in range(max_limit):
                buffer = io.BytesIO()
                img, label = df[i]
                img.save(buffer, 'PNG')
                b64data = base64.b64encode(buffer.getvalue()).decode('utf8')
                data.loc[i] = [b64data, label]

    return data.to_dict(orient='records'), stat
