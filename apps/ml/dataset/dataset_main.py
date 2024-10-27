import io
import base64
import pandas as pd
import sklearn
import torch
import torchvision
import torchaudio
from config.settings import TEMP_DIR

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
        case 'data':
            # convert datetime fields to string
            ts_col = df.select_dtypes(include='datetime').columns.tolist()
            if ts_col is not None:
                for col in ts_col:
                    df[col] = df[col].dt.strftime('%m/%d/%Y %H:%M:%S')

            # get stat info
            stat = df.describe(include='all').T
            stat['type'] = [it.name for it in df.dtypes]
            stat['missing'] = df.isnull().sum().values
            stat_var = df.var(numeric_only=True).to_frame('variance')
            stat = pd.merge(stat, stat_var, left_index=True, right_index=True, how='outer')
            stat = stat.reset_index().rename(columns={"index": "name", "25%": "pct25", "50%": "median", "75%": "pct75"})
            stat = stat.round(3)
            stat = stat.to_dict(orient='records')
            data = df.head(max_limit)
        case 'image':
            # get image attributes as stat info
            img, label = df[0]
            stat_img = dict(name='image', type='int', count=len(df), mode=img.mode, channel=len(img.split()), size=img.size)
            stat_label = dict(name='label', type='int', count=len(df), unique=df.classes)
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
