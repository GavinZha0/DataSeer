import base64
import zlib
import pandas as pd
from apps.ml.eda.chart_plotly import plt_stat_chart, plt_dist_chart, plt_corr_chart, \
    plt_feature_chart, plt_reduction_chart, plt_ts_chart


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

