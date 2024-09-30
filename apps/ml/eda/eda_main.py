import base64
import json
import zlib
import pandas as pd
from pandas import CategoricalDtype
from sqlalchemy.ext.asyncio import AsyncSession
from apps.ml.eda.chart_plotly import plt_stat_chart, plt_dist_chart, plt_corr_chart, \
    plt_feature_chart, plt_reduction_chart, plt_ts_chart
from core.crud import RET
from utils.db_executor import DbExecutor
from apps.datamgr import crud as dm_crud, schema as dm_schemas
from apps.ml import crud as ml_crud, schema as ml_schemas
from sklearn import preprocessing as pp
from sklearn import feature_extraction as fe

"""
return pandas dataframe
"""
async def extract_data(dataset_id: int, db: AsyncSession):
    # get dataset and datasource info
    dataset_info = await ml_crud.DatasetDal(db).get_data(dataset_id, v_ret=RET.SCHEMA)
    source_info = await dm_crud.DatasourceDal(db).get_data(dataset_info.sourceId, v_ret=RET.SCHEMA)

    # connect to target db
    passport = source_info.username + ':' + base64.b64decode(source_info.password).decode('utf-8')
    db = DbExecutor(source_info.type, source_info.url, passport, source_info.params)

    # query data and get dataframe of Pandas
    dataframe, total = await db.db_query(dataset_info.content, None, dataset_info.variable)
    return dataframe, dataset_info.fields, dataset_info.transform

"""
transform data
"""
async def transform_data(df: pd.DataFrame, fields, transform):
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
                df[field_name] = pp.KBinsDiscretizer(n_bins=10, strategy='uniform', encode='ordinal').fit_transform(df[field_name])
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

    return df


"""
return plotly config
"""
async def eda_build_chart(tier: str, kind: str, config, df: pd.DataFrame, fields):
    # use plotly for Pandas plotting
    # pd.options.plotting.backend = "plotly"

    valid_f = [field for field in fields if field.get('omit') is None]
    resp = dict(type='plotly', zip=True, tier=tier, kind=kind, data=None)
    # generate chart based on kind
    match tier:
        case 'stat':
            # Statistics group
            fig = plt_stat_chart(kind, config, df, fields)
        case 'dist':
            # Distribution group
            fig = plt_dist_chart(kind, config, df, valid_f)
        case 'corr':
            # Correlation group
            fig = plt_corr_chart(kind, config, df, valid_f)
        case 'feature':
            # Feature group
            resp['type'] = 'data'
            resp['zip'] = False
            fig = plt_feature_chart(kind, config, df, valid_f)
        case 'dim':
            # Dim reduction group
            fig = plt_reduction_chart(kind, config, df, valid_f)
        case 'ts':
            # Time series group
            fig = plt_ts_chart(kind, config, df, valid_f)
        case _:
            fig = None

    # fig.show()
    if resp.get('zip') and fig is not None:
        figCfgStr = str(fig.to_json())  # don't use json.dumps()
        # zip the config of newPlot()
        zip_data = zlib.compress(figCfgStr.encode())
        # encode data
        resp['data'] = base64.b64encode(zip_data).decode('utf8')
    else:
        resp['data'] = fig

    return resp

