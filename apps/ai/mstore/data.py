import pandas as pd
import requests

from utils.data_loader import DataLoader


async def data_execute(url: str, ds: dict, transform: any = None):
    data = ds
    if transform:
        # initialize data loader without info
        loader = DataLoader(None)
        df = pd.DataFrame(ds.get('dataframe_split')['data'], columns=ds.get('dataframe_split')['columns'])
        trans_df = await loader.transform(df, transform, 'DATA')
        data = {
            'dataframe_split': {
                'data': trans_df.values.tolist(),
                'columns': trans_df.columns.tolist()
            }
        }

    result = requests.post(url=url, json=data)
    result.encoding = 'utf-8'
    return result.json()