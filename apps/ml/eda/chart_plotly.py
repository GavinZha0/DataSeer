import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gluonts.dataset.common import ListDataset
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator
from minisom import MiniSom
from pyod.models import vae, dif, ecod, xgbod, ae1svm, devnet, deep_svdd
from pyod.models.knn import KNN
from sklearn import manifold
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, KernelPCA, FastICA, TruncatedSVD
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sktime.forecasting.timesfm_forecaster import TimesFMForecaster
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.tsa.api import STL
import statsmodels.tsa.seasonal as sm_seasonal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pyod.models.cof import COF
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
import calendar
from scipy.signal import periodogram
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
# import pmdarima as pm
# sktime 0.35.0 requires scikit-learn<1.6.0,>=0.24, but you have scikit-learn 1.6.0 which is incompatible.
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.param_est.seasonality import SeasonalityACF
from sktime.forecasting.trend import PolynomialTrendForecaster
import matplotlib.pyplot as plt
from apps.ml.eda.feature_select import feature_corr_filter, feature_model_eval, feature_iter_search, feature_auto_detect
# from neuralprophet import NeuralProphet
from statsmodels.tsa.arima.model import ARIMA
from sktime.forecasting.fbprophet import Prophet
import ruptures as rpt

"""
build chart of statistics
"""
def plt_stat_chart(kind, config, df, fields):
    # generate chart based on kind
    valid_f = [field for field in fields if field.get('omit') is None]
    match kind:
        case 'overview':
            # overview
            fig = plt_stat_overview(config, df, fields)
        case 'var':
            # Variance
            fig = plt_stat_var(config, df, valid_f)
        case 'box':
            # Boxplot
            fig = plt_stat_box(config, df, valid_f)
        case 'violin':
            # violin plot to show distribution by category
            fig = plt_stat_violin(config, df, valid_f)
        case 'anova':
            # one-way/two-way/multi-factor anova(Analysis of Variance)
            fig = plt_stat_anova(config, df, valid_f)
        case 'outlier':
            # Outliers
            fig = plt_stat_outlier(config, df, valid_f)
        case _:
            # do nothing
            fig = None

    return fig


"""
build chart of distribution
"""
def plt_dist_chart(kind, config, df, fields):
    # generate chart based on kind
    match kind:
        case 'hist':
            # separate histograms with/without kde
            fig = plt_dist_hist(config, df, fields)
        case 'kde':
            # separate Kdes for all numerical columns
            fig = plt_dist_kde(config, df, fields)
        case 'ridge':
            # ridge plot for kde
            fig = plt_dist_ridge(config, df, fields)
        case 'freq':
            fig = plt_dist_freq(config, df, fields)
        case 'kde_2d':
            # 2d kde
            fig = plt_dist_kde2d(config, df, fields)
        case _:
            # do nothing
            fig = None

    return fig

"""
build chart of correlation
"""
def plt_corr_chart(kind, config, df, fields):
    # generate chart based on kind
    match kind:

        case 'scatter':
            # single scatter with margins
            fig = plt_corr_scatter(config, df, fields)
        case 'scatters':
            # scatter matrix
            fig = plt_corr_scatters(config, df, fields)
        case 'pair':
            # pairplot
            fig = plt_corr_pair(config, df, fields)
        case 'corr':
            # Correlation (features to target)
            fig = plt_corr_corr(config, df, fields)
        case 'ccm':
            # Correlation Coefficient Matrix
            fig = plt_corr_ccm(config, df, fields)
        case 'curve':
            if config.get('andrews'):
                # Andrews curves
                fig = plt_corr_andrews(config, df, fields)
            else:
                # Parallel Coordinates curves
                fig = plt_corr_parallel(config, df, fields)
        case _:
            # do nothing
            fig = None

    return fig


"""
dim reduction chart
"""
def plt_reduction_chart(kind, config, df, fields):
    # target field
    cat = None
    t_values = None
    t_field = [field['name'] for field in fields if 'target' in field]
    if t_field is not None:
        f_df = df.drop(columns=t_field)
        for ele in fields:
            # find unique values of category field
            if ele['name'] == t_field[0] and ele['attr'] == 'cat':
                t_values = ele.get('values')
                cat = ele['name']
                break

    dim = 2  # keep all dims by default
    if config.get('dim'):
        dim = config['dim']

    if dim <= 0:
        # auto select dims
        dim = None
    elif dim > len(f_df.columns):
        # reduce 1 dim at lest
        dim = len(f_df.columns)

    neighbor = 5
    if config.get('neighbor'):
        neighbor = config['neighbor']

    title = None
    labels = {str(i): kind.upper() + f"{i}" for i in range(dim)}
    fig = go.Figure()
    match kind:
        case 'pca':
            if config.get('kernel'):
                # Kernel PCA (linear, poly, rbf, sigmoid, cosine)
                # 先将数据从原空间映射到高维空间，然后在特征空间进行PCA
                # 引入核函数解决非线性数据映射问题,计算协方差矩阵时使用了核函数
                # 无监督，非线性
                pca = KernelPCA(n_components=dim, kernel=config['kernel'])
                data = pca.fit_transform(f_df)
            else:
                # Principal Component Analysis(主成分分析)
                # 目标是向数据变化最大的方向投影,或者说向重构误差最小化的方向投影。
                # 假设:数据的主要成分是方差的体现
                # PCA适用于找到数据中的主要成分，降低数据的冗余性,常用于数据预处理和特征提取
                # 无监督，线性
                # n_components = None， 不降维，返回各个原始特征的方差比例
                # n_components = 0.96（小数），总的解释方差比例
                # n_components = 5（整数），想要的维数
                # n_components = 'mle'，自动选择
                pca = PCA(n_components=dim)
                data = pca.fit_transform(f_df)
                total_var_pct = pca.explained_variance_ratio_.sum() * 100
                title = f'Total Explained Variance: {total_var_pct:.2f}%'
                labels = {
                    str(i): f"PCA{i} ({var:.1f}%)"
                    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
                }
        case 'ica':
            # Independent Component Analysis(独立成分分析)
            # 通过对数据进行线性变换，使得变换后的数据各个分量间尽可能地独立
            # ICA 是使每个分量最大化独立，便于发现隐藏因素
            # ICA通常不用于降低维度，而用于分离叠加信号,在信号处理、图像处理、语音识别等领域有广泛应用
            # 无监督，非线性
            # algorithm: ‘parallel’, ‘deflation’
            algo = 'parallel'
            if config.get('algo'):
                algo = config['algo']
            ica = FastICA(n_components=dim, algorithm=algo)
            data = ica.fit_transform(f_df)
        case 'svd':
            # Singular Value Decomposition(奇异值分解)
            # 线性降维技术，它将数据方差较小的特征投影到低维空间
            # 广泛应用于线性代数、信号处理和机器学习等领域
            # 无监督，线性
            if dim < 2:
                return None
            svd = TruncatedSVD(n_components=dim)
            data = svd.fit_transform(f_df)
        case 'tsne':
            # t-distributed Stochastic Neighbor Embedding( t-分布随机近邻嵌入)
            # 种以数据原有的趋势为基础，重建其在低纬度(二维或三维)下数据趋势,流形不变,能很好地保留数据全局结构
            # 无监督，非线性
            # perplexity: [5, 50]
            perplex = 30
            if config.get('perplex'):
                perplex = config['perplex']
            pca = manifold.TSNE(n_components=dim, perplexity=perplex, init='pca')
            data = pca.fit_transform(f_df)
        case 'isomap':
            # isometric mapping (等度量映射算法)
            # 解决MDS算法在非线性结构数据集上的弊端,数据在向低维空间映射之后能够保持流形不变
            # 常用于手写数字等数据的降维
            # 无监督，非线性
            pca = manifold.Isomap(n_components=dim, n_neighbors=neighbor)
            data = pca.fit_transform(f_df)
        case 'lle':
            # locally linear embedding(局部线性嵌入算法)
            # 将高维数据投影到低维空间中，使其保持数据点之间的局部线性重构关系，即有相同的重构系数, 流形不变
            # 无监督，非线性
            # method: ‘standard’, ‘hessian’, ‘modified’ or ‘ltsa’
            algo = 'standard'
            if config.get('algo'):
                algo = config['algo']
            pca = manifold.LocallyLinearEmbedding(n_components=dim, n_neighbors=neighbor, method=algo)
            data = pca.fit_transform(f_df)
        case 'lda':
            # Linear Discriminant Analysis(线性判别分析,用于分类问题)
            # 目标是向最大化类间差异，最小化类内差异的方向投影, 以利于分类等任务即将不同类的样本有效的分开.
            # 有监督，线性
            if cat is None:
                return None

            if t_values is None:
                t_values = df[cat].unique()
            if t_values:
                threshold = min(len(f_df.columns), len(t_values) - 1)
                if dim > threshold:
                    dim = min(dim, threshold)
            pca = LinearDiscriminantAnalysis(n_components=dim)
            data = pca.fit_transform(f_df, df[cat])
            total_var_pct = pca.explained_variance_ratio_.sum() * 100
            title = f'Total Explained Variance: {total_var_pct:.2f}%'
            labels = {
                str(i): f"LDA{i} ({var:.1f}%)"
                for i, var in enumerate(pca.explained_variance_ratio_ * 100)
            }
        case 'autocode':
            # Auto Encoder
            # 神经网络的一种，它是一种无监督算法
            # 自动编码器通过消除重要特征上的噪声和冗余，找到数据在较低维度的表征。
            '''
            from keras.models import Model
            from keras.layers import Input, Dense
            # Fixed dimensions
            input_dim = data.shape[1]  # 8
            encoding_dim = 3
            # Number of neurons in each Layer [8, 6, 4, 3, ...] of encoders
            input_layer = Input(shape=(input_dim,))
            encoder_layer_1 = Dense(6, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
            encoder_layer_2 = Dense(4, activation="tanh")(encoder_layer_1)
            encoder_layer_3 = Dense(encoding_dim, activation="tanh")(encoder_layer_2)
            # Crear encoder model
            encoder = Model(inputs=input_layer, outputs=encoder_layer_3)
            # Use the model to predict the factors which sum up the information of interest rates.
            encoded_data = pd.DataFrame(encoder.predict(data_scaled))
            encoded_data.columns = ['factor_1', 'factor_2', 'factor_3']
            '''
            keras_needed = True
        case '_':
            # do nothing
            return None

    # convert 2 dim array to dataframe
    dim_df = pd.DataFrame(data)
    # add target field
    dim_df[t_field] = df[t_field]

    if t_values and df[cat].dtype == 'category':
        # replace codes to category names
        dim_df[cat] = dim_df[cat].cat.rename_categories(t_values)

    if dim == 1:
        fig = px.scatter(x=dim_df.index, y=dim_df[0], color=None if cat is None else dim_df[cat], labels=labels, title=title)
    elif dim == 2:
        fig = px.scatter(dim_df, x=0, y=1, color=None if cat is None else dim_df[cat], labels=labels, title=title)
    elif dim == 3:
        fig = px.scatter_3d(dim_df, x=0, y=1, z=2, size=[5]*len(dim_df), size_max=10, labels=labels, title=title,
                            color=None if cat is None else dim_df[cat])
    else:
        fig = go.Figure(go.Table(header=dict(values=[kind + f'{i}' for i in range(dim)]),
                                 cells=dict(values=np.transpose(np.round(data, 3)))))
    return fig


"""
build chart of feature
"""
def plt_feature_chart(kind, config, df, fields):
    # get target field
    t_field = [field['name'] for field in fields if 'target' in field]
    t_df = df[t_field]
    # get date field
    ts_date_field = [field['name'] for field in fields if field['attr'] == 'date' and field.get('timeline')]
    if len(ts_date_field):
        # set datetime field as index if it is time series
        # df.set_index(ts_date_field[0], inplace=True)
        # feature df includes target field if it is time series
        f_df = df
    else:
        f_df = df[df.columns.difference(t_field)]

    match kind:
        case 'corrfilter':
            data = feature_corr_filter(f_df, t_df, config, None if len(ts_date_field)==0 else ts_date_field[0])
        case 'modeleval':
            data = feature_model_eval(f_df, t_df, config, None if len(ts_date_field)==0 else ts_date_field[0])
        case 'itersearch':
            data = feature_iter_search(f_df, t_df, config, None if len(ts_date_field)==0 else ts_date_field[0])
        case 'autodetect':
            data = feature_auto_detect(f_df, t_df, config, None if len(ts_date_field)==0 else ts_date_field[0])
        case _:
            # do nothing
            data = None
    return data

"""
build chart of reduction
"""
def plt_ts_chart(kind, config, df, fields):
    ts_field = None
    date_fields = [field['name'] for field in fields if field['attr'] == 'date']
    ts_fields = df.select_dtypes(include='datetime').columns.tolist()

    if config.get('tf'):
        # selected time field
        ts_field = config['tf']
    elif len(date_fields):
        # the first specific date field
        ts_field = date_fields[0]
    elif len(ts_fields):
        # the first date field which detected by Pandas
        ts_field = ts_fields[0]
    else:
        # no ts field
        return None

    if config.get('vf'):
        if config.get('cat'):
            df = df[[ts_field, config['cat'], config['vf']]]
        else:
            df = df[[ts_field, config['vf']]]
    # set ts field as index
    df.set_index(ts_field, inplace=True)
    if len(df.columns) == 1:
        config['vf'] = df.columns[0]
    match kind:
        case 'series':
            # time series lines
            fig = plt_ts_series(ts_field, config, df, fields)
        case 'trend':
            # Trending line
            fig = plt_ts_trend(ts_field, config, df, fields)
        case 'diff':
            # difference curve
            fig = plt_ts_diff(ts_field, config, df, fields)
        case 'compare':
            # Comparison
            fig = plt_ts_compare(ts_field, config, df, fields)
        case 'mavg':
            # Moving Average
            fig = plt_ts_mavg(ts_field, config, df, fields)
        case 'tsfreq':
            # quarterly, monthly, weekly and daily
            fig = plt_ts_freq(ts_field, config, df, fields)
        case 'autocorr':
            # ACF/PACF
            fig = plt_ts_acf(ts_field, config, df, fields)
        case 'quantile':
            # box and violin
            fig = plt_ts_quantile(ts_field, config, df, fields)
        case 'cycle':
            # Periodicity detection
            fig = plt_ts_cycle(ts_field, config, df, fields)
        case 'decomp':
            # decomposition(trend + season)
            fig = plt_ts_decomp(ts_field, config, df, fields)
        case 'predict':
            # prediction
            fig = plt_ts_predict(ts_field, config, df, fields)
        case 'anomaly':
            # anomaly detection
            fig = plt_ts_anomaly(ts_field, config, df, fields)
        case 'anc':
            # active noice reduction
            fig = plt_ts_anc(ts_field, config, df, fields)
        case _:
            fig = None

    return fig

"""
Record Overview
"""
def plt_stat_overview(cfg, df, fields):
    na_rows = df.isnull().T.any().sum()
    dup_rows = df.duplicated().sum()
    total_rows = len(df)
    pie_names = ['Valid', 'Missing', 'Duplicate']
    pie_values = [total_rows - na_rows - dup_rows, na_rows, dup_rows]

    # attr fields
    valid_f = [field for field in fields if field.get('omit') is None]
    attr_names = ['Continuous', 'Discrete', 'Category', 'Datetime', 'Coordinate']
    attr_values = [len([field for field in valid_f if field['attr'] == 'conti']),
                   len([field for field in valid_f if field['attr'] == 'disc']),
                   len([field for field in valid_f if field['attr'] == 'cat']),
                   len([field for field in valid_f if field['attr'] == 'date']),
                   len([field for field in valid_f if field['attr'] == 'coord'])]

    # rows with missing value
    miss_r = df.isnull().sum(axis=1)
    miss_r = miss_r[miss_r.values > 0]
    # columns with missing value
    miss_df = df.isnull().sum().to_frame()
    miss_df.reset_index(inplace=True, names=['name'])
    miss_df.rename(columns={0: 'miss'}, inplace=True)

    num_fields = [field['name'] for field in valid_f if field['attr'] == 'conti'] + \
                 [field['name'] for field in valid_f if field['attr'] == 'disc']

    coff = 1.6
    # get statistics info
    stat = df[num_fields].describe()
    # Inter Quantile Range
    iqr = stat.loc['75%'] - stat.loc['25%']
    th_l = stat.loc['25%'] - iqr * coff
    th_u = stat.loc['75%'] + iqr * coff
    outers = df[num_fields][(df[num_fields] < th_l) | (df[num_fields] > th_u)]
    outers = outers.dropna(axis=0, how='all').dropna(axis=1, how='all')
    outer_df = outers.count().to_frame()
    outer_df.reset_index(inplace=True, names=['name'])
    outer_df.rename(columns={0: 'outer'}, inplace=True)
    miss_outer = pd.merge(miss_df, outer_df, how='left')

    fig = make_subplots(rows=2, cols=3, specs=[[{}, {'type': 'domain'}, {}], [{"colspan": 3}, None, None]],
                        subplot_titles=['Field count by Attribute', 'Missing Rate',
                                        'Missing histogram by Count', 'Missing & Outlier'],
                        horizontal_spacing=0.02, vertical_spacing=0.1)
    fig.add_trace(go.Bar(x=attr_names, y=attr_values, text=attr_values, name=''), row=1, col=1)
    fig.add_trace(go.Pie(labels=pie_names, values=pie_values, name='', textinfo='label+percent', hole=.5,
                         textposition='inside', hoverinfo="label+value"), row=1, col=2)
    fig.add_trace(go.Histogram(x=miss_r.values, name='', texttemplate="%{y}"), row=1, col=3)

    fig.add_trace(go.Bar(x=miss_outer['name'], y=miss_outer['miss'], name='Missing', text=miss_outer['miss']),
                  row=2, col=1)
    fig.add_trace(go.Bar(x=miss_outer['name'], y=miss_outer['outer'], name='Outlier', text=miss_outer['outer']),
                  row=2, col=1)

    fig.update_yaxes(row=1, col=1, tickformat=",d")
    fig.update_xaxes(row=1, col=3, tickformat=",d")
    fig.update_yaxes(row=1, col=3, tickformat=",d")
    fig.update_yaxes(row=2, col=1, tickformat=",d")
    fig.update_layout(showlegend=False, hovermode=None)
    fig.add_annotation(text=total_rows, x=.5, y=.79, font_size=18, showarrow=False, xref="paper", yref="paper")
    return fig


"""
Box plot for numeric fields
离群值是指低于Q1-1.5*IQR或高于Q3+1.5*IQR的数据(IQR为Q3-Q1)。
"""
def plt_stat_box(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']
    sorted_name = df[[n for n in num_fields]].mean(numeric_only=True).sort_values().index

    # show mean by default
    boxmean = True
    if cfg.get('sd'):
        # represent mean and standard deviation
        boxmean = 'sd'

    outlier = False
    if cfg.get('outlier'):
        outlier = 'outliers'

    # category field
    cf = None
    if cfg.get('cf'):
        cf = cfg['cf']

    # quantilemethod: linear, inclusive, exclusive
    fig = go.Figure()
    if cf is None or cf == '':
        for name in sorted_name:
            fig.add_trace(go.Box(y=df[name], name=name, boxmean=boxmean, boxpoints=outlier))
    else:
        for name in sorted_name:
            c_values = df[cf]
            for f in fields:
                if f['name'] == cf and f.get('values'):
                    c_values = df[cf].map(dict(enumerate(f['values'])))
                    break
            fig.add_trace(go.Box(x=c_values, y=df[name], name=name, boxmean=boxmean, boxpoints=outlier))
        fig.update_layout(boxmode='group')

    return fig

"""
Violin plot for numeric fields
"""
def plt_stat_violin(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']
    sorted_name = df[[n for n in num_fields]].mean(numeric_only=True).sort_values().index

    box = False
    if 'box' in cfg:
        box = cfg['box']

    outlier = False
    if cfg.get('outlier'):
        outlier = 'outliers'

    # category field
    cf = None
    if cfg.get('cf'):
        cf = cfg['cf']

    # quantilemethod: linear, inclusive, exclusive
    fig = go.Figure()
    if cf is None or cf == '':
        for name in sorted_name:
            fig.add_trace(go.Violin(y=df[name], name=name, points=outlier, box_visible=box, meanline_visible=True))
    else:
        for name in sorted_name:
            c_values = df[cf]
            for f in fields:
                if f['name'] == cf and f.get('values'):
                    c_values = df[cf].map(dict(enumerate(f['values'])))
                    break
            fig.add_trace(go.Violin(x=c_values, y=df[name], name=name, points=outlier, box_visible=box, meanline_visible=True))
        fig.update_layout(violinmode='group')

    return fig


"""
ANOVA: Analysis of Variance (方差分析，变异分析, or F test)
检验某feature的多个分组数据（by a cat field）的均值是否存在显著差异
如果pValue<0.05，说明存在显著差异(各数值特征存在显著差异; 即对结果有显著影响)
ANOVA分析前需要满足3个假设: 每组样本具备方差同质性、组内样本服从正态分布，样本间需要独立。
One-way ANOVA,单因素分析，顾名思义就是分析单一因素在组间的差异(不同颜色的饮料销量比较，单一因素颜色，目标销量差异)
Multiple-factor(include two-way) ANOVA,多(双)因素分析，分析两个及以上分类特征对一个数值域的影响哪个更大(两个分类特征形成交叉表)
组内方差: 各个数值型特征各自的方差
组间方差：各数值型特征针对某一分类特征（或分类目标）之间的方差
ANOVA目的是检验每个组的平均数是否相同
"""
def plt_stat_anova(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']
    cat_fields = [field['name'] for field in fields if field['attr'] == 'cat']
    fig = go.Figure()
    # calculate all numeric fields based on selected category field (1 cat: one-way, 2cats: two-way)
    # calculate all discrete fields based on selected continuous field
    if cfg.get('field') is None:
        return fig

    sel_field = cfg['field']
    sel_attr = None
    sel_values = None
    for ele in fields:
        if ele['name'] == sel_field:
            sel_attr = ele['attr']
            sel_values = ele.get('values')
            break

    if sel_attr == 'cat':
        # one-way anova based on selected category field
        # every numeric field is a single variable
        # 某一特征的不同分类数据间的差异分析,然后汇总所有特征的分析结果并绘制在一起
        cat_list = df[sel_field].value_counts().index.tolist()
        cat_names = df[sel_field].value_counts().index.tolist()
        if sel_values and df[sel_field].dtype == 'category':
            # get category names for displaying if it is category type
            cat_names = sel_values

        title = 'One-way ANOVA based on category field ' + sel_field
        num_col = 3
        num_row = math.ceil(len(num_fields) / num_col)
        fig = make_subplots(rows=num_row, cols=num_col,
                            subplot_titles=['{}'.format(n) for n in num_fields],
                            row_heights=[400 for i in range(num_row)],
                            horizontal_spacing=0.05, vertical_spacing=0.2 / num_row)
        for idx, name in enumerate(num_fields):
            # mean and std of every category group
            mean_std = df.groupby(by=sel_field, observed=True)[name].agg(['mean', 'std']).reset_index()
            # the data of every category group
            cat_data = [df[df[sel_field] == c][name] for c in cat_list]
            # get p value then insert to mean_std
            # F = MSB/MSE(组间方差与组内方差的比率，对比F分布统计表，如果大于临界值，说明总体均值间存在差异)
            # pValue<0.05，说明分类数据间存在显著差异
            f_v, p_v = stats.f_oneway(*cat_data)
            # mean_std.insert(2, 'pvalue', np.round(p_v, 3))
            if cfg.get('line'):
                fig.add_trace(go.Scatter(x=cat_names, y=mean_std['mean'].round(3), name='',
                                         error_y=dict(type='data', array=mean_std['std'].round(3))),
                              (idx // num_col) + 1, (idx % num_col) + 1)
            else:
                # mean-std chart by categories for selected numeric field
                fig.add_trace(go.Bar(x=cat_names, y=mean_std['mean'].round(3), name='',
                                     error_y=dict(type='data', array=mean_std['std'].round(3))),
                              (idx // num_col) + 1, (idx % num_col) + 1)
            # update subplot title to add p-value
            fig.layout.annotations[idx].text = '{} (p={})'.format(name, np.round(p_v, 3))
        fig.update_layout(title=title, showlegend=False)
    elif sel_attr == 'conti' or sel_attr == 'disc':
        # multiple-factor anova (包括Two-way ANOVA)
        # 分析两个及以上分类特征对一个数值特征的影响程度
        # 模型的公式为“y ~ A + B + C + A*B + A*C + B*C + A*B*C”，
        # y是因变量，A、B和C是自变量。A:B、A:C和B:C是自变量的交互作用。
        formula = sel_field + '~'
        for name in cat_fields:
            formula = formula + '+' + name
        ana = smf.ols(formula, data=df).fit()
        fp_df = sm.stats.anova_lm(ana, type=2).dropna(how='any')
        # % percentage (<5%: 显著相关)
        fp_df.sort_values(by='PR(>F)', inplace=True)
        fp_df['PR(>F)'] = (fp_df['PR(>F)'] * 100).round(5)
        fig.add_trace(go.Bar(x=fp_df.index, y=fp_df['PR(>F)']))
    fig.update_xaxes(type='category')
    return fig

"""
Outlier detection
PyOD(Python Outlier Detection):用于检测数据中异常值的库，它能对20多种不同的算法进行访问
"""
def plt_stat_outlier(cfg, df, fields):
    valid_fields = [field['name'] for field in fields if 'omit' not in field]
    target_field = [field['name'] for field in fields if 'target' in field]
    feature_fields = list(set(valid_fields).difference(set(target_field)))
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc' and 'target' not in field]
    fig = go.Figure()

    method = 'quantile'
    if cfg.get('method'):
        method = cfg['method']

    metric = 'euclidean'
    if cfg.get('metric'):
        metric = cfg['metric']

    # contamination: 默认为0.05，即5%的异常值
    # irq_coff: 默认为1.6，即1.6倍IQR
    # sigma_coff: 默认为3，即3倍标准差
    # use same parameter name for all methods
    threshold = None
    if cfg.get('threshold'):
        threshold = cfg['threshold']

    y_pred = None
    match method:
        case 'quantile':
            # distribution-based
            iqr_coff = threshold if threshold else 1.6

            # get statistics info
            # IQR: InterQuartile range (四分位距)
            stat = df[num_fields].describe()
            # Inter Quantile Range
            iqr = stat.loc['75%'] - stat.loc['25%']
            th_l = stat.loc['25%'] - iqr * iqr_coff
            th_u = stat.loc['75%'] + iqr * iqr_coff
            outers = df[num_fields][(df[num_fields] < th_l) | (df[num_fields] > th_u)]
            outers = outers.dropna(axis=0, how='all').dropna(axis=1, how='all')

            num_col = 4
            num_row = math.ceil(len(outers.columns) / num_col)
            if num_row == 0:
                return None
            fig = make_subplots(rows=num_row, cols=num_col,
                                subplot_titles=['{} ({})'.format(n, outers[n].count()) for n in outers.columns],
                                row_heights=[400 for n in range(num_row)],
                                horizontal_spacing=0.05, vertical_spacing=0.2 / num_row)
            for idx, name in enumerate(outers):
                # show precomputed box and outliers
                fig.add_trace(go.Box(y=[outers[name].to_list()], name='', notched=False, pointpos=0,
                                     q1=[stat.loc['25%', name]], median=[stat.loc['50%', name]],
                                     q3=[stat.loc['75%', name]], lowerfence=[th_l[name]],
                                     upperfence=[th_u[name]], line_color='#21deb8', marker_color='#db4052', marker_outliercolor='#db4052'),
                              (idx // num_col) + 1, (idx % num_col) + 1)
                # show plotly box
                # fig.add_trace(go.Box(x=df[name], name="", quartilemethod=algo, line_color='#6baed6', notched=True,
                #                      marker=dict(outliercolor='#db4052')),
                #               (idx // num_col) + 1, (idx % num_col) + 1)
            fig.update_xaxes(type='category')
            fig.update_layout(height=num_row * 400, showlegend=False)
            return fig
        case 'zscore':
            # distribution-based
            sigma_coff = threshold if threshold else 3

            # get statistics info
            stat = df[num_fields].describe()
            # 3 sigma line
            n3 = (stat.loc['mean'] - stat.loc['std'] * sigma_coff).round(3)
            p3 = (stat.loc['mean'] + stat.loc['std'] * sigma_coff).round(3)
            # outliers
            outer_l = df[num_fields][df[num_fields] < n3]
            outer_u = df[num_fields][df[num_fields] > p3]
            outer_l = outer_l.dropna(axis=0, how='all')
            outer_l['type'] = 'lower'
            outer_u = outer_u.dropna(axis=0, how='all')
            outer_u['type'] = 'upper'
            outers = pd.concat([outer_l, outer_u])
            outers = outers.dropna(axis=1, how='all')

            num_col = 3
            num_row = math.ceil((len(outers.columns)-1) / num_col)
            if num_row == 0:
                return None

            fig = make_subplots(rows=num_row, cols=num_col,
                                subplot_titles=['{} ({})'.format(n, outers[n].count()) for n in outers.columns if n != 'type'],
                                row_heights=[400 for n in range(num_row)],
                                horizontal_spacing=0.05, vertical_spacing=0.2 / num_row)
            for idx, name in enumerate(outers.columns):
                if name == 'type':
                    continue
                # kde
                df_no_na = df[name].dropna()
                vls = df_no_na.value_counts().index.to_list()
                if len(df_no_na) < 2 or len(vls) < 2:
                    # less than 2 data points
                    continue

                kder = stats.gaussian_kde(df_no_na)
                x_range = df[name].max() - df[name].min()
                # generate 10 points between min and max at least
                kde_x = np.linspace(df[name].min()-abs(df[name].min())*0.1, df[name].max()+abs(df[name].max())*0.1, max(math.ceil(x_range * 10), 30), True)
                kde_y = kder.evaluate(kde_x)
                # Keep 5 decimal places
                kde_x = np.round(kde_x, 5)
                kde_y = np.round(kde_y, 5)
                fig.add_trace(go.Scatter(x=kde_x, y=kde_y, name='', fill='tozeroy'), (idx // num_col) + 1,
                              (idx % num_col) + 1)

                lower_x = []
                upper_x = []
                if len(outer_l[name]) > 0:
                    lower_x = outer_l[name].to_list()
                lower_y = np.linspace(0.1, kde_y.max(), len(outer_l[name]))
                if len(outer_u[name]) > 0:
                    upper_x = outer_u[name].to_list()
                upper_y = np.linspace(0.1, kde_y.max(), len(outer_u[name]))

                fig.add_trace(go.Scatter(x=lower_x+upper_x, y=lower_y.tolist() + upper_y.tolist(), name='', mode='markers'),
                              (idx // num_col) + 1, (idx % num_col) + 1)

                # add -3/+3 std dev lines
                fig.add_shape(editable=False, type="line", x0=n3[name], x1=n3[name], y0=0, y1=kde_y.max(),
                              line=dict(dash="dot"), row=(idx // num_col) + 1, col=(idx % num_col) + 1)
                fig.add_shape(editable=False, type="line", x0=p3[name], x1=p3[name], y0=0, y1=kde_y.max(),
                              line=dict(dash="dot"), row=(idx // num_col) + 1, col=(idx % num_col) + 1)
            fig.update_layout(height=400 * num_row, showlegend=False, hovermode='x')
            return fig
        case 'dbscan':
            # Density-Based Spatial Clustering of Applications with Noise，具有噪声的基于密度的聚类方法(cluster-based)
            #  ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’,
            #  ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
            #  ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’
            # 'minkowski' does not work
            distance = threshold if threshold else 0.5
            clf = DBSCAN(eps=distance, metric=metric)
            clf.fit(df[num_fields])
            y_pred = [1 if i < 0 else 0 for i in clf.labels_]
        case 'svm':
            kernel = 'rbf'
            if cfg.get('kernel'):
                kernel = cfg['kernel']
            # One-Class SVM (classfication-based)
            # kernel: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
            cont_ratio = threshold if threshold else 0.05
            clf = OCSVM(nu=0.5, contamination=cont_ratio, kernel=kernel)
            clf.fit(df[num_fields])
            y_pred = clf.labels_
        case 'knn':
            #  K-Nearest Neighbors (distance-based)
            # ['braycurtis', 'canberra', 'chebyshev',
            # 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
            # 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
            # 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
            # 'sqeuclidean', 'yule']
            # 'correlation' does not work
            # contamination: proportion of outliers
            cont_ratio = threshold if threshold else 0.05
            clf = KNN(contamination=cont_ratio, n_neighbors=5, method='mean', metric=metric)
            clf.fit(df[num_fields])
            # 1: outlier
            y_pred = clf.labels_
        case 'lof':
            # Local Outlier Factor(局部利群因子, density-based)
            # ['braycurtis', 'canberra', 'chebyshev',
            # 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
            # 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
            # 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
            # 'sqeuclidean', 'yule']
            cont_ratio = threshold if threshold else 0.05
            clf = LOF(contamination=cont_ratio, n_neighbors=10, metric=metric)
            y_pred = clf.fit_predict(df[num_fields])
        case 'cof':
            # Connectivity-Based Outlier Factor (COF, LOF的变种, density-based)
            cont_ratio = threshold if threshold else 0.05
            clf = COF(contamination=cont_ratio, n_neighbors=15)
            clf.fit(df[num_fields])
            y_pred = clf.predict(df[num_fields])
        case 'iforest':
            # Isolation Forest(孤立森林, tree-based)
            # unsupervised, global outlier detection
            # contamination: percentage of outliers
            # n_estimators: total of trees
            cont_ratio = threshold if threshold else 0.05
            clf = IForest(contamination=cont_ratio, n_estimators=100, max_samples='auto')
            clf.fit(df[num_fields])
            y_pred = clf.predict(df[num_fields])
        case 'som':
            # self-organizing map(自组织映射算法)
            # 是一种无监督学习算法，用于对高维数据进行降维和聚类分析
            # 无监督，非线性
            cont_ratio = threshold if threshold else 0.05
            som = MiniSom(15, 15, len(num_fields), sigma=3, learning_rate=0.5, neighborhood_function='triangle')
            som.train_batch(df[num_fields].values, 1000)
            # 量化误差（Quantization Error）,计算每个样本到最近神经元（BMU, Best Matching Unit）的距离
            quantization_errors = np.linalg.norm(som.quantization(df[num_fields].values) - df[num_fields].values, axis=1)
            # 基于分位数, x% outliers
            error_treshold = np.percentile(quantization_errors, (1-cont_ratio)*100)
            # outlier is True
            is_outlier = quantization_errors > error_treshold
            # 1: outlier
            y_pred = is_outlier.astype(int)

            # 基于Z-Score, 3 omiga
            # z_scores = np.abs(stats.zscore(quantization_errors))
            # is_outlier = z_scores > 3
            # y_pred = is_outlier.astype(int)

        case 'vae':
            # AutoEncoder(自编码器, unsupervised, neural network)
            cont_ratio = threshold if threshold else 0.05
            auto_encoder = vae.VAE(epoch_num=50, batch_size=32, contamination=cont_ratio)
            auto_encoder.fit(df[num_fields])
            # 1: outlier
            y_pred = auto_encoder.predict(df[num_fields])
        case '_':
            return fig

    # display outliers by t-SNE/UMAP chart
    dim = 2
    if cfg.get('d3'):
        dim = 3

    if len(num_fields)==1:
        # time series with one value field
        date_fields = [field['name'] for field in fields if field['attr'] == 'date' or 'timeline' in field]
        df.set_index(date_fields[0], inplace=True)
        if len(date_fields) > 0:
            y_df = pd.DataFrame(df[num_fields[0]]*y_pred)
            y_df = y_df[y_df[num_fields[0]]>0]
            fig.add_trace(go.Scatter(x=df.index, y=df[num_fields[0]], name=num_fields[0], hovertemplate='%{y}<extra></extra>'))
            fig.add_trace(go.Scatter(x=y_df.index, y=y_df[num_fields[0]], name='outlier', line=dict(color="#ff0000"),
                                     mode='markers', hovertemplate='%{y}<extra></extra>'))
            fig.update_layout(hovermode='x')
            fig.update_layout(title=f'total outliers: {len(y_df)}', hovermode='x')
        return fig
    else:
        dim = min(dim, len(num_fields))

    # visualization solution of 2d/3d
    if cfg.get('umap'):
        aa = 55
        # UMAP: Uniform Manifold Approximation and Projection
        # data = umap.UMAP(n_components=dim).fit_transform(df[num_fields])
    else:
        # t-distributed Stochastic Neighbor Embedding
        data = manifold.TSNE(n_components=dim).fit_transform(df[num_fields])

    vis_df = pd.DataFrame(data)
    vis_df['type'] = ['inner' if i == 0 else 'outlier' for i in y_pred]
    title = f'Method: {method}, Outliers: {sum(y_pred)}'
    # discrete color
    # cmap = [[((i + 1) // 2) / 2, px.colors.qualitative.Plotly[i // 2]] for i in range(2 * 2)]
    if dim == 3:
        fig = px.scatter_3d(vis_df, x=0, y=1, z=2, size=[5]*len(vis_df), color=vis_df['type'], size_max=10, opacity=1,
                            color_discrete_map={"inner": "#00CC96", "outlier": "#EF553B"},
                            category_orders={'type': ['inner', 'outlier']}, title=title)
    else:
        fig = px.scatter(vis_df, x=0, y=1, color='type', title=title,
                         color_discrete_map={"inner": "#00CC96", "outlier": "#EF553B"},
                         category_orders={'type': ['inner', 'outlier']})

    fig.update_layout(title=title, xaxis_title='', yaxis_title='', legend_title_text='',
                      legend=dict(xanchor="right", yanchor="top", x=0.99, y=0.99))
    return fig


"""
Variance rank
"""
def plt_stat_var(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']

    vars = df[num_fields].var().sort_values().round(3)
    fig = px.bar(x=vars.index.to_list(), y=vars.values, title='Variance Rank', text_auto=True)
    fig.update_layout(hovermode='x')
    return fig

"""
histogram for numeric fields with/without kde
"""
def plt_dist_hist(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']

    num_col = 3
    num_row = math.ceil(len(num_fields) / num_col)
    # enable secondary y axes for kde
    fig_specs = [[{"secondary_y": True} for i in range(num_col)] for j in range(num_row)]
    fig = make_subplots(rows=num_row, cols=num_col, specs=fig_specs,
                        subplot_titles=[n for n in num_fields], row_heights=[400 for n in range(num_row)],
                        horizontal_spacing=0.05, vertical_spacing=0.2/num_row)

    bins = 30
    if cfg.get('bins'):
        bins = cfg['bins']

    for idx, name in enumerate(num_fields):
        if cfg.get('cf'):
            # selected field should be a category field
            for seq, val in enumerate(df[cfg['cf']].value_counts().index):
                fig.add_trace(go.Histogram(x=df[df[cfg['cf']] == val][name], name='',
                                           nbinsx=bins, marker_color=px.colors.qualitative.Plotly[seq], opacity=0.8),
                              (idx // num_col) + 1, (idx % num_col) + 1, secondary_y=False)
        else:
            fig.add_trace(go.Histogram(x=df[name], name='', nbinsx=bins, opacity=0.8),
                          (idx // num_col) + 1, (idx % num_col) + 1, secondary_y=False)
        if cfg.get('kde'):
            # add kde curve to histogram
            df_without_na = df[name].dropna()
            vls = df_without_na.value_counts().index.to_list()
            if len(df_without_na) < 2 or len(vls) < 2:
                continue
            kder = stats.gaussian_kde(df_without_na)
            x_range = math.ceil(df[name].max()) - math.floor(df[name].min())
            kde_x = np.linspace(math.floor(df[name].min()), math.ceil(df[name].max()), max(x_range * 15, 50), True)
            kde_y = kder.evaluate(kde_x)
            # Keep 3 decimal places
            kde_x = np.round(kde_x, 5)
            kde_y = np.round(kde_y, 5)
            fig.add_trace(go.Scatter(x=kde_x, y=kde_y, name='', line=dict(color='#c9dd22')), (idx // num_col) + 1,
                          (idx % num_col) + 1, secondary_y=True)
    fig.update_layout(height=400 * num_row, showlegend=False, hovermode='x', barmode='stack')
    return fig


"""
KDE for numeric fields with/without reference curve
"""
def plt_dist_kde(cfg, df, fields):
    num_col = 3
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']

    # combined kde chart
    if cfg.get('combine'):
        rug = False
        if cfg.get('rug'):
            rug = True

        fig = ff.create_distplot([df[n] for n in num_fields], [n for n in num_fields], show_hist=False, show_rug=rug)
        return fig

    # build kde lines
    ax = ff.create_distplot([df[n] for n in num_fields], [n for n in num_fields], show_hist=False, show_rug=False)

    # separate kde charts
    ref = cfg.get('ref')
    num_row = math.ceil(len(num_fields) / num_col)
    fig = make_subplots(rows=num_row, cols=num_col, subplot_titles=[n for n in num_fields],
                        row_heights=[400 for n in range(num_row)],
                        horizontal_spacing=0.05, vertical_spacing=0.2/num_row)

    for idx, name in enumerate(num_fields):
        fig.add_trace(go.Scatter(x=ax.data[idx].x, y=ax.data[idx].y, name='', fill='tozeroy'), (idx // num_col) + 1,
                      (idx % num_col) + 1)
        if ref:
            # extended range
            x_range = math.ceil(df[name].max()) - math.floor(df[name].min())
            ref_x = np.linspace(math.floor(df[name].min()), math.ceil(df[name].max()), max(x_range * 20, 100), True)
            match ref:
                case 'norm':
                    # norm distribution (gaussian distribution)
                    ref_name = 'Norm'
                    ref_y = stats.norm(loc=df[name].mean(), scale=df[name].std()).pdf(ref_x)
                case 'log':
                    # Lognormal distribution
                    ref_name = 'Log'
                    ref_y = stats.lognorm(loc=df[name].mean(), s=df[name].std()).pdf(ref_x)
                case 'exp':
                    # Exponential distribution
                    ref_name = 'Exp'
                    ref_y = stats.expon(loc=df[name].mean(), scale=df[name].std()).pdf(ref_x)
                case 'lap':
                    # Exponential distribution
                    ref_name = 'Laplace'
                    ref_y = stats.laplace(loc=df[name].mean(), scale=df[name].std()).pdf(ref_x)
                case _:
                    # do nothing
                    continue
            # add reference curve
            ref_x = np.round(ref_x, 5)
            ref_y = np.round(ref_y, 5)
            fig.add_trace(go.Scatter(x=ref_x, y=ref_y, name=ref_name, line=dict(color='#b68100', dash='dot')),
                          (idx // num_col) + 1, (idx % num_col) + 1)
    fig.update_layout(height=400 * num_row, showlegend=False, hovermode='x')
    return fig


"""
ridge plot for numeric fields
"""
def plt_dist_ridge(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']

    # sort fields based on mean for displaying
    cf_name = df[[n for n in num_fields]].mean(numeric_only=True).sort_values().index
    fig = go.Figure()
    for name in cf_name:
        fig.add_trace(go.Violin(x=df[name], name=name))

    if len(num_fields) < 10:
        fig_h = 150*len(num_fields)
    else:
        fig_h = 100 * len(num_fields)
    fig.update_traces(orientation='h', side='positive', width=3, points=False)
    fig.update_layout(height=fig_h, xaxis_zeroline=False)
    return fig


"""
calculate the frequency of categorical fields
"""
def plt_dist_freq(cfg, df, fields):
    num_col = 2
    row_h = 400
    cat_fields = [field for field in fields if field['attr'] == 'cat']
    num_row = math.ceil(len(cat_fields) / num_col)
    if num_row == 0:
        return go.Figure()

    if cfg.get('funnel'):
        # 漏斗图
        num_col = 3
        row_h = 600

    pct = False
    if cfg.get('pct'):
        pct = True

    fig = make_subplots(rows=num_row, cols=num_col, subplot_titles=[c['name'] for c in cat_fields],
                        row_heights=[row_h for n in range(num_row)],
                        horizontal_spacing=0.05, vertical_spacing=0.2 / num_row)

    for idx, field in enumerate(cat_fields):
        name = field['name']
        cat_df = df[name].value_counts().reset_index()
        if field.get('values') and cat_df[name].dtype == 'category':
            cat_df[name] = cat_df[name].cat.rename_categories(field.get('values'))
        cat_df.sort_values(by='count', inplace=True)
        cat_df.reset_index(drop=True, inplace=True)
        if pct:
            cat_df['pct'] = round(100*cat_df['count']/cat_df['count'].sum(), 1)
            if cfg.get('funnel'):
                fig.add_trace(go.Funnel(x=cat_df['count'], y=cat_df[name], name=name, textinfo='percent total'),
                              (idx // num_col) + 1, (idx % num_col) + 1)
            else:
                fig.add_trace(go.Bar(x=cat_df[name], y=cat_df['count'], name=name, text=cat_df['pct'].astype(str) + '%'),
                              (idx // num_col) + 1, (idx % num_col) + 1)
        else:
            if cfg.get('funnel'):
                fig.add_trace(go.Funnel(x=cat_df['count'], y=cat_df[name], name=name, textinfo='value'),
                              (idx // num_col) + 1, (idx % num_col) + 1)
            else:
                fig.add_trace(go.Bar(x=cat_df[name], y=cat_df['count'], name=name, text=cat_df['count']),
                              (idx // num_col) + 1, (idx % num_col) + 1)
        fig.layout.annotations[idx].text = '{} ({})'.format(name, len(cat_df))
    if cfg.get('funnel'):
        # avoid float on y axes
        fig.update_yaxes(type='category')

    fig.update_layout(height=row_h*num_row, showlegend=False, hovermode='x', barmode='stack')
    return fig


"""
calculate the frequency of categorical fields
"""
def plt_freq_old(cfg, df, fields):
    disc_fields = [field for field in fields if field['attr'] == 'cat']
    num_col = 2
    num_row = math.ceil(len(disc_fields) / num_col)
    fig = make_subplots(rows=num_row, cols=num_col, subplot_titles=[c['name'] for c in disc_fields],
                        row_heights=[400 for n in range(num_row)], horizontal_spacing=0.05,
                        vertical_spacing=0.2/num_row)
    for idx, field in enumerate(disc_fields):
        name = field['name']
        rank = df[name].value_counts().to_frame()
        # hue should be true for category display
        if 'field' in cfg and cfg['field']:
            for seq, u_value in enumerate(df[cfg['field']].value_counts().index.to_list()):
                rank = df[df[cfg['field']] == u_value][name].value_counts().to_frame()
                fig.add_trace(go.Bar(x=rank.index.tolist(), y=rank.values.T[0].tolist(), name=u_value),
                              (idx // num_col) + 1, (idx % num_col) + 1)
        else:
            fig.add_trace(go.Bar(x=rank.index.tolist(), y=rank.values.T[0].tolist(), name=name, text=rank.values.T[0].tolist()),
                          (idx // num_col) + 1, (idx % num_col) + 1)
    if cfg.get('order') == 'asc':
        fig.update_xaxes(type='category', categoryorder='total ascending')
    else:
        fig.update_xaxes(type='category', categoryorder='total descending')
    fig.update_layout(height=400 * num_row, showlegend=False, hovermode='x', barmode='stack')
    return fig



"""
Single Scatter
"""
def plt_corr_scatter(cfg, df, fields):
    if cfg.get('xf') is None or cfg.get('yf') is None:
        return None

    xf = cfg['xf']
    yf = cfg['yf']

    marg = None  # histogram, rug, box, violin
    if cfg.get('marg'):
        marg = cfg['marg']

    facet = None  # facet_row
    if cfg.get('facet'):
        facet = cfg['facet']

    frac = None  # osl, rolling, ewm, expanding, lowess
    trend = None
    if cfg.get('frac'):
        frac = cfg['frac']
        trend = 'lowess'

    cat = None
    if cfg.get('cf'):
        cat = cfg['cf']
        if df[cat].dtype == 'category':
            for ele in fields:
                # find target unique values
                if ele['name'] == cat:
                    t_values = ele.get('values')
                    if t_values:
                        # replace codes to category names
                        df[cat] = df[cat].cat.rename_categories(t_values)
                    break

    if cat:
        # category by colors
        fig = px.scatter(df, x=xf, y=yf, color=cfg['cf'], trendline=trend, trendline_scope="overall",
                         trendline_options=dict(frac=frac), facet_row=facet, marginal_x=marg, marginal_y=marg)
    else:
        fig = px.scatter(df, x=xf, y=yf, trendline=trend, trendline_options=dict(frac=frac),
                         facet_row=facet, marginal_x=marg, marginal_y=marg)

    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    return fig


"""
Scatter Matrix without upperhalf and diagonal
"""
def plt_corr_scatters(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                  [field['name'] for field in fields if field['attr'] == 'disc']

    cat = None
    if cfg.get('cf'):
        cat = cfg['cf']
        if df[cat].dtype == 'category':
            for ele in fields:
                # find target unique values
                if ele['name'] == cat:
                    t_values = ele.get('values')
                    if t_values:
                        # replace codes to category names
                        df[cat] = df[cat].cat.rename_categories(t_values)
                    break

    fig = px.scatter_matrix(df, dimensions=num_fields, color=cat)
    fig.update_traces(showupperhalf=False, diagonal_visible=False)
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    return fig


"""
Pair plot
"""
def plt_corr_pair(cfg, df, fields):
    # show scatter matrix based on selected dimensions
    pair_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']
    cat = None
    box = 'histogram'  # histogram, box
    if cfg.get('box'):
        box = 'box'
    if cfg.get('cf'):
        cat = cfg['cf']
        if df[cat].dtype == 'category':
            for ele in fields:
                # find target unique values
                if ele['name'] == cat:
                    t_values = ele.get('values')
                    if t_values:
                        # replace codes to category names
                        df[cat] = df[cat].cat.rename_categories(t_values)
                    break

        pair_fields.append(cat)

    if len(pair_fields) > 1:
        fig = ff.create_scatterplotmatrix(df[pair_fields], diag=box, index=cat)
        fig.update_layout(title='', width=1400, height=700)
        return fig
    else:
        return None



"""
CCM: Correlation Coefficient Matrix
"""
def plt_corr_ccm(cfg, df, fields):
    # 相关系数也可以看成协方差：一种剔除了两个变量量纲影响、标准化后的特殊协方差。
    # Pearson 系数用来检测两个连续型变量之间线性相关程度，要求这两个变量分别分布服从正态分布；
    # Spearman系数不假设连续变量服从何种分布，如果是顺序变量(Ordinal)，推荐使用Spearman。
    # Kendall 用于检验连续变量和类别变量间的相关性
    # 卡方检验(Chi-squared Test)，检验类别变量间的相关性
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']
    cat_fields = [field['name'] for field in fields if field['attr'] == 'cat']
    coeff = 'pearson'  # pearson, kendall, spearman
    if cfg.get('coeff'):
        coeff = cfg['coeff']
    if cfg.get('num'):
        df_corr = df[[n for n in num_fields]].corr(method=coeff, numeric_only=True).round(2)
    else:
        df_corr = df[[n for n in num_fields+cat_fields]].corr(method=coeff).round(2)

    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    viz_corr = df_corr.mask(mask).dropna(how='all').dropna(axis=1, how='all')
    viz_corr = viz_corr.replace({np.nan: ''})

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=viz_corr, x=viz_corr.columns, y=viz_corr.index,
                             hoverinfo="none", colorscale=px.colors.diverging.RdBu, text=viz_corr.values,
                             texttemplate="%{text}", zmin=-1, zmax=1, ygap=1, xgap=1))
    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, yaxis_autorange='reversed', template='plotly_white')
    return fig



"""
Parallel curves
"""
def plt_corr_parallel(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']
    # cfg['group'] = 'disc'  # disc or conti
    # map target fields to discrete colors
    cat = None
    t_values = []
    if cfg.get('cf'):
        cat = cfg['cf']
        if df[cat].dtype == 'category':
            for ele in fields:
                # find target unique values
                if ele['name'] == cat:
                    t_values = ele.get('values')
                    break

        cat_field_values = df[cat].value_counts().index
        cmap = [[((i + 1) // 2) / len(cat_field_values), px.colors.qualitative.Plotly[i // 2]] for i in
                range(len(cat_field_values) * 2)]

        fig = px.parallel_coordinates(df, dimensions=num_fields, color=cat, color_continuous_scale=cmap)
        # show color bar with target names
        fig.update_layout(coloraxis_colorbar=dict(title=None, tickvals=np.arange(len(t_values)), ticktext=t_values))
    else:
        fig = px.parallel_coordinates(df)

    fig.update_layout(height=700)
    return fig


"""
Andrews curve
"""
def plt_corr_andrews(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']

    fig = go.Figure()
    cat = 'fake_cat'
    if cfg.get('cf'):
        cat = cfg['cf']
        if df[cat].dtype == 'category':
            for ele in fields:
                # find target unique values
                if ele['name'] == cat:
                    t_values = ele.get('values')
                    if t_values:
                        # replace codes to category names
                        df.sort_values(by=cat, inplace=True)
                        colors = df[cat].to_list()
                        df[cat] = df[cat].cat.rename_categories(t_values)
                    break
        num_fields.append(cat)
    else:
        num_fields.append(cat)
        df[cat] = [0]*len(df)
        colors = [0]*len(df)

    # build andrews curve by pandas.plotting
    plt.figure()
    axes = pd.plotting.andrews_curves(df[num_fields], cat, samples=100)
    # get curve data from pandas chart then build plotly chart
    lines = axes.get_lines()
    for idx, line in enumerate(lines):
        data = line.get_data()
        legendshow = True
        c_idx = 0
        if idx == 0:
            legendshow = True
        elif colors[idx] != colors[idx-1]:
            c_idx = c_idx + 1
            legendshow = True
        else:
            legendshow = False
        if cfg.get('cf'):
            fig.add_trace(go.Scatter(x=data[0], y=data[1], name=df.loc[idx, cat], legendgroup=df.loc[idx, cat],
                                     showlegend=legendshow, line=dict(color=px.colors.qualitative.Plotly[colors[idx]])))
        else:
            fig.add_trace(go.Scatter(x=data[0], y=data[1], name='', line=dict(color=px.colors.qualitative.Plotly[0]),
                                     showlegend=False))
    fig.update_layout(height=700, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    return fig


"""
Correlation between features and target
"""
def plt_corr_corr(cfg, df, fields):
    method = 'pearson'  # pearson, kendall, spearman
    num_only = False
    t_field = None
    fig = go.Figure()
    if cfg.get('method'):
        method = cfg['method']
    if cfg.get('num_only'):
        num_only = cfg['num_only']
    if cfg.get('field'):
        t_field = cfg['field']
        corrs = df.drop(columns=[t_field]).corrwith(df[t_field], method=method, numeric_only=num_only).abs()
        corrs.sort_values(ascending=False, inplace=True)
        corrs = corrs.to_frame()
        corrs = corrs.dropna(how='any').round(3)
        fig = corrs.plot.bar(text_auto=True)
        fig.update_layout(title='Correlation between features and target (' + t_field + ')',
                          xaxis_title='', yaxis_title='', width=1400, height=800,
                          showlegend=False, hovermode=False)

    return fig






"""
return pandas dataframe
"""
def plt_dist_kde2d(cfg, df, fields):
    fig = px.density_heatmap(df, x="sepal_length", y="sepal_width", marginal_x="box", marginal_y="violin")
    return fig


"""
plt ts trending chart
{"pid": "ts", "tf": "time", "period": "YE", "agg": "mean", "solo": true, "connected": true}
"""
def plt_ts_overview(tsf, cfg, df, fields):
    fig = go.Figure()
    range_min = df.index.min()
    range_max = df.index.max()
    duration = range_max - range_min
    resolution = df.index.resolution
    diff = df.index.diff()
    gap_min = diff.min().total_seconds()
    gap_max = diff.max().total_seconds()


"""
plt ts trending chart
{"pid": "ts", "tf": "time", "period": "YE", "vf": "kpi", "agg": "mean", "cat": "country", "solo": true, "connected": true}
"""
def plt_ts_series(tsf, cfg, df, fields):
    fig = go.Figure()
    v_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']

    cat_fields = [field['name'] for field in fields if field['attr'] == 'cat']

    if cfg.get('vf'):
        v_fields = [cfg['vf']]

    # agg: sum, mean, median, min, max, std, count
    agg = 'mean'  # default
    if cfg.get('agg'):
        agg = cfg['agg']

    connected = True
    if cfg.get('connected'):
        connected = cfg['connected']

    agg_cfg = {}
    for nf in v_fields:
        agg_cfg[nf] = agg

    # category field
    cat_field = cfg.get('cat')

    period = 'D'
    if cfg.get('period'):
        ts_df, period = ts_resample(tsf, cfg, df, fields)
        '''
        period = cfg['period']
        # aggregated by period (YS,QS,MS,W,D,h,min,s)
        ts_df = df.resample(cfg['period']).agg(agg_cfg)
        if period.startswith('Y'):
            # show year only (don't show month and day for Y)
            ts_df.index = ts_df.index.strftime('%Y')
        elif period.startswith('QQ'):
            # convert to period for getting quarters
            ts_df.index = ts_df.index.to_period('Q')
            ts_df.index = ts_df.index.strftime('%Y-Q%q')
        '''
    else:
        cat_fields.insert(0, tsf)
        ts_df = df.groupby(cat_fields).agg(agg_cfg)
        ts_df.reset_index(inplace=True)
        ts_df.set_index(tsf, inplace=True)

    if cfg.get('solo'):
        # put curves on separated charts
        rows = len(v_fields)
        fig = make_subplots(rows=rows, cols=1, subplot_titles=v_fields, row_heights=[300 for n in range(rows)],
                            horizontal_spacing=0.05, vertical_spacing=0.2 / rows)
        [fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[nf], name='', connectgaps=connected,
                                  hovertemplate='%{y}<extra></extra>'), i + 1, 1) for i, nf in enumerate(v_fields)]
        fig.update_xaxes(matches='x')
        fig.update_layout(showlegend=False, height=rows * 300, hovermode='x')
    else:
        if cat_field:
            if ts_df[cat_field].nunique() > 50:
                page = cfg['page'] if cfg.get('page') else 0
                cat_values = ts_df[cat_field].unique()
                cat_values.sort()
                cat_50 = cat_values[page*50:(page+1)*50]
                ts_df = ts_df[ts_df[cat_field].isin(cat_50)]
            ts_df.reset_index(inplace=True)
            # df.sort_values(by=[tsf, cat_field], inplace=True)
            fig = px.line(ts_df, x=tsf, y=v_fields[0], color=cat_field)
            # fig.update_layout(legend_traceorder="normal")
        else:
            # put all curves on one chart
            [fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[nf], name=nf, connectgaps=connected,
                                      hovertemplate='%{y}<extra></extra>')) for nf in v_fields]
            fig.update_layout(hovermode='x')

    return fig


"""
plt ts trending line chart
# ols(线性普通最小二乘), lowess(局部加权线性回归), rolling(移动平均线), ewm(指数加权移动平均), expanding(扩展窗)
{"pid": "ts", "ts": "date", "vf": "open",  "period": "M", "agg": "mean", "frac": 0.6}
"""
def plt_ts_trend(tsf, cfg, df, fields):
    # value field
    if cfg.get('vf') is None or cfg.get('period') is None:
        return None

    vfield = cfg['vf']
    connected = False
    if cfg.get('connected'):
        connected = cfg['connected']

    # aggregated by period (YS,QS,MS,W,D,h,min,s)
    ts_df, period = ts_resample(tsf, cfg, df, fields)
    if period.startswith('Y'):
        # show year only (don't show month and day for Y)
        ts_df.index = ts_df.index.strftime('%Y')
    elif period.startswith('QQ'):
        # convert to period for getting quarters
        ts_df.index = ts_df.index.to_period('Q')
        ts_df.index = ts_df.index.strftime('%Y-%q')

    max_win = len(ts_df)
    frac = 0.667  # [0, 1]
    win = 5
    if cfg.get('frac'):
        frac = cfg['frac']
        if frac < 0:
            frac = 0
        elif frac > 1:
            frac = 1

    win = math.floor(frac * max_win)
    if win <= 0:
        win = 1

    fig = go.Figure()
    # ts_df.reset_index(inplace=True)
    # 线性
    fig = px.scatter(ts_df, x=ts_df.index, y=vfield, trendline='ols')
    ts_df['ols'] = fig.data[1].y

    # The fraction of the data used when estimating each y-value.
    # 平滑
    fig = px.scatter(ts_df, x=ts_df.index, y=vfield, trendline='lowess', trendline_options=dict(frac=frac))
    ts_df['lowess'] = fig.data[1].y

    # 中心滞后，权重相同
    fig = px.scatter(ts_df, x=ts_df.index, y=vfield, trendline='rolling', trendline_options=dict(window=win, min_periods=1))
    ts_df['rolling'] = fig.data[1].y

    # 中心滞后，权重衰减
    fig = px.scatter(ts_df, x=ts_df.index, y=vfield, trendline='ewm', trendline_options=dict(halflife=win))
    ts_df['ewm'] = fig.data[1].y

    # it is OLS when degree=1
    forecaster = PolynomialTrendForecaster(degree=int(1//frac))
    prange = pd.date_range(ts_df.index.min(), periods=len(ts_df), freq=period)
    ts_df['polynomial'] = forecaster.fit(ts_df[vfield]).predict(fh=prange)

    # time series line
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[vfield], name=vfield, connectgaps=connected))
    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df['ols'], name='Ols'))
    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df['lowess'], name='Lowess'))
    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df['rolling'], name='Rolling'))
    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df['ewm'], name='Ewm'))
    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df['polynomial'], name='Polynomial'))
    fig.update_layout(hovermode='x')
    return fig


"""
plt ts difference chart
# 差分，与前面diff个周期值的差，可见指标增长或下降
{"pid": "ts", "ts": "time", "period": "YE", "agg": "mean", "solo": false, "field": "dena72", "diff": 1, "step": 1}
"""
def plt_ts_diff(tsf, cfg, df, fields):
    fig = go.Figure()
    if cfg.get('period') is None:
        return fig

    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']
    diff = 1
    if cfg.get('diff'):
        diff = cfg['diff']

    step = 1
    if cfg.get('step'):
        step = cfg['step']

    # agg: sum, mean, median, min, max, std, count
    agg = 'mean'  # default
    if cfg.get('agg'):
        agg = cfg['agg']

    vf = None
    agg_cfg = {}
    if cfg.get('vf'):
        vf = cfg['vf']
        agg_cfg[vf] = agg
    else:
        for nf in num_fields:
            agg_cfg[nf] = agg

    # aggregated by period (YS,QS,MS,W,D,h,min,s)
    ts_df, period = ts_resample(tsf, cfg, df, fields)

    if period.startswith('Y'):
        # show year only (don't show month and day for Y)
        ts_df.index = ts_df.index.strftime('%Y')

    if vf:
        # specific value field
        for i in range(step):
            ts_df[vf] = ts_df[vf].diff(periods=diff)
        fig.add_trace(go.Bar(x=ts_df.index, y=ts_df[vf], name=vf))
        fig.update_layout(height=800, hovermode='x')
    else:
        for i in range(step):
            ts_df[num_fields] = ts_df[num_fields].diff(periods=diff)
        if cfg.get('solo'):
            # put curves on separated charts
            rows = len(num_fields)
            fig = make_subplots(rows=rows, cols=1, subplot_titles=num_fields, row_heights=[300 for n in range(rows)],
                                 horizontal_spacing=0.05, vertical_spacing=0.2/rows)
            [fig.add_trace(go.Bar(x=ts_df.index, y=ts_df[nf], name=''), i+1, 1) for i, nf in enumerate(num_fields)]
            fig.update_xaxes(matches='x')
            fig.update_layout(showlegend=False, height=rows*300, hovermode='x')
        else:
            # put all curves on one chart
            [fig.add_trace(go.Bar(x=ts_df.index, y=ts_df[nf], name=nf, hovertemplate='%{y}<extra></extra>'))
             for nf in num_fields]
            fig.update_layout(hovermode='x')
    return fig


"""
plt ts frequency chart
尽量选择一个完整的周期，如2020-08-05 到2-24-08-05
{"pid": "ts", "ts": "time", "field": "dena74", "agg": "sum"}
"""
def plt_ts_freq(tsf, cfg, df, fields):
    fig = go.Figure()
    if cfg.get('vf') is None:
        return fig

    # agg: sum, mean, median, min, max, count
    agg = 'mean'
    vfield = cfg['vf']  # metrix field
    if cfg.get('agg'):
        agg = cfg['agg']

    min_ts = df.index.min()
    max_ts = df.index.max()
    title = 'Datetime from ' + min_ts.strftime('%Y-%m-%d') + ' to ' + max_ts.strftime('%Y-%m-%d')
    fig = make_subplots(rows=3, cols=2, specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
                        subplot_titles=['Quarterly', 'Monthly', 'Weekly', 'Hourly', 'Daily'])

    # Quarterly
    # aaaa = df.resample('Q').asfreq()
    ts_df = pd.DataFrame({tsf: df.index.quarter, vfield: df[vfield]})
    ts_df.set_index(tsf, inplace=True)
    gp_df = ts_df.groupby(tsf).agg(agg).round(3)
    per_df = pd.DataFrame(columns=[tsf], data=range(1, 5, 1))
    merged_df = pd.merge(per_df, gp_df, left_on=tsf, right_index=True, how="left")
    fig.add_trace(
        go.Bar(x='Q' + merged_df[tsf].astype('string'), y=merged_df[vfield], name='', text=merged_df[vfield]), 1, 1)

    # Monthly
    ts_df = pd.DataFrame({tsf: df.index.strftime('%b'), vfield: df[vfield]})
    ts_df.set_index(tsf, inplace=True)
    gp_df = ts_df.groupby(tsf).agg(agg).round(3)
    per_df = pd.DataFrame(columns=[tsf], data=pd.date_range(start='2024-01-01', end='2024-12-01', freq='MS')
                          .strftime('%b').to_list())
    merged_df = pd.merge(per_df, gp_df, left_on=tsf, right_index=True, how="left")
    fig.add_trace(go.Bar(x=merged_df[tsf], y=merged_df[vfield], name='', text=merged_df[vfield]), 1, 2)

    # Weekly
    ts_df = pd.DataFrame({tsf: df.index.strftime('%a'), vfield: df[vfield]})
    ts_df.set_index(tsf, inplace=True)
    gp_df = ts_df.groupby(tsf).agg(agg).round(3)
    per_df = pd.DataFrame(columns=[tsf],
                          data=pd.date_range(start='2024-09-01', end='2024-09-07', freq='D')
                          .strftime('%a').to_list())
    merged_df = pd.merge(per_df, gp_df, left_on=tsf, right_index=True, how="left")
    fig.add_trace(go.Bar(x=merged_df[tsf], y=merged_df[vfield], name='', text=merged_df[vfield]), 2, 1)

    # Hourly
    ts_df = pd.DataFrame({tsf: df.index.hour, vfield: df[vfield]})
    ts_df.set_index(tsf, inplace=True)
    gp_df = ts_df.groupby(tsf).agg(agg).round(3)
    per_df = pd.DataFrame(columns=[tsf], data=range(0, 24, 1))
    merged_df = pd.merge(per_df, gp_df, left_on=tsf, right_index=True, how="left")
    fig.add_trace(go.Bar(x=merged_df[tsf], y=merged_df[vfield], name='', text=merged_df[vfield]), 2, 2)

    # Daily
    ts_df = pd.DataFrame({tsf: df.index.day, vfield: df[vfield]})
    ts_df.set_index(tsf, inplace=True)
    gp_df = ts_df.groupby(tsf).agg(agg).round(3)
    per_df = pd.DataFrame(columns=[tsf], data=range(1, 32, 1))
    merged_df = pd.merge(per_df, gp_df, left_on=tsf, right_index=True, how="left")
    fig.add_trace(go.Bar(x=merged_df[tsf], y=merged_df[vfield], name='', text=merged_df[vfield]), 3, 1)

    fig.update_xaxes(type='category')
    fig.update_layout(title=title, showlegend=False)
    return fig


"""
plt ts compare chart
# Y: year, q: quarter, m: month, w: week, d: day, H: hour, M: min, S: sec. 
# refer to https://pandas.pydata.org/docs/reference/api/pandas.Period.strftime.html
{"pid": "ts", "ts": "time", "field": "dena74", "group": "m", "period": "Y", "agg": "sum"}
"""
def plt_ts_compare(tsf, cfg, df, fields):
    fig = go.Figure()
    if cfg.get('vf') is None or cfg.get('groupby') is None or cfg.get('period') is None:
        return fig

    # agg: sum, mean, median, min, max, count
    agg = 'mean'
    if cfg.get('agg'):
        agg = cfg['agg']

    vfield = cfg['vf']
    group = cfg['groupby']
    period = cfg['period']

    if group == 'q':
        # q is not available in strftime()
        df['ts_' + group] = df.index.quarter
    else:
        # all these are same and numbers
        df['ts_' + group] = df.index.strftime('%' + group)

    if period == 'q':
        # q is not available in strftime()
        df['ts_' + period] = df.index.quarter
    else:
        # all these are same and numbers
        df['ts_' + period] = df.index.strftime('%' + period)

    ts_df = df.groupby(['ts_' + group, 'ts_' + period]).agg(agg).round(3)
    ts_df.reset_index(inplace=True)
    # convert to number for sort and map
    ts_df[['ts_' + group, 'ts_' + period]] = ts_df[['ts_' + group, 'ts_' + period]].apply(pd.to_numeric)
    ts_df.sort_values(by=['ts_' + group, 'ts_' + period], inplace=True)

    # convert number to name
    if group == 'm':
        mm = dict(enumerate(calendar.month_abbr))
        ts_df['ts_' + group] = ts_df['ts_' + group].map(mm)
    elif group == 'w':
        ww = dict(enumerate(calendar.day_abbr))
        ts_df['ts_' + group] = ts_df['ts_' + group].map(ww)

    if period == 'm':
        mm = dict(enumerate(calendar.month_abbr))
        ts_df['ts_' + period] = ts_df['ts_' + period].map(mm)
    elif period == 'w':
        ww = dict(enumerate(calendar.day_abbr))
        ts_df['ts_' + period] = ts_df['ts_' + period].map(ww)

    # why didn't I use hisFunc for agg - Gavin??
    fig = px.histogram(ts_df, x='ts_' + group, y=vfield, color='ts_' + period, barmode='group')
    fig.update_xaxes(type='category')
    fig.update_layout(xaxis_title='', yaxis_title='', legend_title='')
    return fig


"""
plt ts ACF/PACF chart
# 可用于判断时间序列是否为平稳序列
# 平稳要求序列数据不能有趋势、不能有周期性
单调序列：ACF衰减到0的速度很慢，而且可能一直为正，或一直为负，或先正后负，或先负后正。
周期序列：ACF呈正弦波动规律。
平稳序列：ACF衰减到0的速度很快，并且十分靠近0，并控制在2倍标准差内。
根据ACF/PACF确定ARIMA模型p，q值: 如果说自相关图拖尾，并且偏自相关图在p阶截尾时，此模型应该为AR(p)。如果说自相关图在q阶截尾并且偏自相关图拖尾时，此模型应该为MA(q)。
如果说自相关图和偏自相关图均显示为拖尾，那么可结合ACF图中最显著的阶数作为q值，选择PACF中最显著的阶数作为p值，最终建立ARMA(p,q)模型。
如果说自相关图和偏自相关图均显示为截尾，那么说明不适合建立ARIMA模型。
{"pid": "ts", "ts": "time", "field": "dena74", "period": "m", "agg": "sum", "lag": 20}
"""
def plt_ts_acf(tsf, cfg, df, fields):
    fig = go.Figure()
    if cfg.get('period') is None or cfg.get('vf') is None:
        return fig

    # group by date field and period(YE,Q,M,W,D,H,min,s) then aggregate numerical fields
    # agg: sum, mean, median, min, max, count
    vfield = cfg['vf']
    ts_df, period = ts_resample(tsf, cfg, df, fields)

    lag = 10
    if cfg.get('lag'):
        lag = cfg['lag']
    pacf_limit = math.floor(len(ts_df) / 2)
    if lag > pacf_limit:
        lag = pacf_limit - 1

    # adf test and kpss test to detect if it is stationary
    adf_test = adfuller(ts_df[vfield], autolag='AIC')
    p_val1 = adf_test[1]
    kpss_test = kpss(ts_df[vfield], regression='c')
    p_val2 = kpss_test[1]

    test_result = 'non-stationary'
    if (p_val1 < 0.05) and (p_val2 > 0.05):
        test_result = 'stationary'

    ts_df.reset_index(inplace=True)
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[f'Autocorrelation ({test_result})', f'Partial Autocorrelation ({test_result})'])
    # Auto Correlation Function(自相关)
    acf_df = acf(ts_df[vfield], nlags=lag)
    acf_df = acf_df.round(3)
    [fig.add_trace(go.Scatter(x=(x, x), y=(0, acf_df[x]), mode='lines', name='', line_color='#3f3f3f'), 1, 1)
     for x in range(len(acf_df))]
    fig.add_trace(go.Scatter(x=np.arange(len(acf_df)), y=acf_df, name='', mode='markers'), 1, 1)

    # partial Auto Correlation Function(偏自相关)
    # method: ols, yw, ywm, ld, ldb
    pacf_df = pacf(ts_df[vfield], nlags=lag)
    pacf_df = pacf_df.round(3)
    [fig.add_trace(go.Scatter(x=(x, x), y=(0, pacf_df[x]), mode='lines', name='', line_color='#3f3f3f'), 2, 1)
     for x in range(len(pacf_df))]
    fig.add_trace(go.Scatter(x=np.arange(len(pacf_df)), y=pacf_df, name='', mode='markers'), 2, 1)

    fig.update_layout(showlegend=False)
    return fig


"""
plt ts Moving Average chart
# 如果窗口大小为整数，那窗口在数据列上从前向后平移，每次移动一格(period)计算指定窗口大小数量的数据而忽略时间间隔
# 如果窗口大小为时间间隔(如'1D', '3M')，那窗口在时间列上从前先后移动，每次移动一格并计算给定时间窗口内的数据
SMA：权重系数一致；WMA：权重系数随时间间隔线性递减；EMA：权重系数随时间间隔指数递减。
# it is trend line when win==1
# window type: https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows
{"pid": "ts", "ts": "time", "field": "dena74", "period": "m", "agg": "sum", "win": 3}
"""
def plt_ts_mavg(tsf, cfg, df, fields):
    fig = go.Figure()
    if cfg.get('vf') is None:
        return fig

    vfield = cfg['vf']
    win = 3
    if cfg.get('win'):
        win = cfg['win']

    # group by date field and period(YS,QS,MS,W,D,H,min,s) then aggregate numerical fields
    # agg data based on period first
    ts_df, period = ts_resample(tsf, cfg, df, fields)

    min_per = win
    if isinstance(win, int):
        min_per = math.ceil(win/2)
    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[vfield], name=vfield, connectgaps=True))
    ts_df[vfield] = ts_df[vfield].rolling(window=win, min_periods=min_per).mean()
    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[vfield], name='SMA', connectgaps=True))
    if isinstance(win, int):
        # WMA and EMA don't support date win like '2M'
        ts_df[vfield] = ts_df[vfield].rolling(window=win).apply(lambda x: x[::-1].cumsum().sum() * 2 / win / (win + 1))
        fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[vfield], name='WMA', connectgaps=True))

        ts_df[vfield] = ts_df[vfield].ewm(span=win).mean()
        fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[vfield], name='EMA', connectgaps=True))

    return fig


"""
plt ts box/violin chart
{"pid": "ts", "ts": "time", "field": "dena74", "period": "Q", "violin": true}
"""
def plt_ts_quantile(tsf, cfg, df, fields):
    fig = go.Figure()
    if cfg.get('period') is None or cfg.get('vf') is None:
        return fig

    period = cfg['period']
    vfield = cfg['vf']  # metrix field
    df.index = df.index.to_period(period)
    df.reset_index(inplace=True)
    ts_df = df.groupby(tsf)
    if period.startswith('Y'):
        ts_format = '%Y'
    elif period.startswith('Q'):
        ts_format = '%YQ%q'
    elif period.startswith('M'):
        ts_format = '%Y-%b'
    elif period.startswith('W') or period.startswith('us'):
        ts_format = '%Y-W%U'
    elif period.startswith('D'):
        ts_format = '%Y-%m-%d'
    elif period.startswith('H'):
        ts_format = '%Y-%m-%d %H'
    elif period.startswith('min'):
        ts_format = '%Y-%m-%d %H:%M'
    else:
        ts_format = '%Y-%m-%d %H:%M:%S'

    for freq, data in ts_df:
        if len(data) > 0 or (len(data) == 0 and cfg.get('gap')):
            if cfg.get('violin'):
                fig.add_trace(go.Violin(y=data[vfield], name=freq.strftime(ts_format), points='outliers',
                                        box_visible=True, meanline_visible=False))
            else:
                fig.add_trace(go.Box(y=data[vfield], name=freq.strftime(ts_format), boxmean=True,
                                     boxpoints='outliers'))
    return fig


"""
plt ts cycle chart
{"pid": "ts", "ts": "date", "field": "open",  "period": "M", "agg": "mean", "algo": "psd"}
"""
def plt_ts_cycle(tsf, cfg, df, fields):
    fig = go.Figure()
    if cfg.get('vf') is None:
        return fig

    vfield = cfg['vf']
    # aggregated by period (YS,QS,MS,W,D,h,min,s)
    ts_df, period = ts_resample(tsf, cfg, df, fields)
    if period is not None and period.startswith('Y'):
        # show year only (don't show month and day for Y)
        ts_df.index = ts_df.index.strftime('%Y')
    elif period is not None and period.startswith('Q'):
        # convert to period for getting quarters
        ts_df.index = ts_df.index.to_period('Q')
        ts_df.index = ts_df.index.strftime('%Y-Q%q')

    algo = 'psd'
    if cfg.get('algo'):
        algo = cfg['algo']

    ts_df.reset_index(inplace=True)
    match algo:
        case 'psd':
            # Periodogram(周期图), PSD(Power spectral density, 功率谱密度)
            # 傅里叶变换和频谱分析
            freq, power = periodogram(ts_df[vfield])
            freq = np.round(freq, 4)
            power = np.round(power, 4)

            psd_df = pd.DataFrame({'freq': freq, 'power': power})
            psd_df = psd_df.sort_values(by='power', ascending=False)
            # get top 5 by power
            top_power = psd_df.head(5)
            top_power = top_power[top_power['power'] > 0]
            # get cycles
            top_power['cycle'] = np.round(1 / top_power['freq'], 1)

            # sort by index
            psd_df.sort_index(inplace=True)
            top_power.sort_index(inplace=True)
            # fig = px.line(x=freq, y=power, labels={'x': 'Freq', 'y': 'Power'})
            fig.add_trace(
                go.Scatter(x=psd_df['freq'], y=psd_df['power'], name='PSD', showlegend=False, hovertemplate='%{y}<extra></extra>'))
            fig.add_trace(
                go.Scatter(x=top_power['freq'], y=top_power['power'], text=top_power['cycle'], name='Cycle',
                           mode='markers+text', textposition='top right', hovertemplate='%{text}<extra></extra>'))
            fig.update_xaxes(title='Freq')
            fig.update_yaxes(title='Power')
            fig.update_layout(hovermode='x')
        case '_':
            return fig

    return fig


"""
plt ts decomposition chart
加法模型：y（t）=季节+趋势+周期+噪音
乘法模型：y（t）=季节*趋势*周期*噪音
# doesn't support 'min' and 's'
{"pid": "ts", "ts": "time", "field": "dena74",  "period": "D", "agg": "mean", "algo": "stl", "robust": true}
"""
def plt_ts_decomp(tsf, cfg, df, fields):
    fig = go.Figure()
    if cfg.get('vf') is None or cfg.get('period') is None:
        return fig

    vfield = cfg['vf']
    # aggregated by period (YS,QS,MS,W,D,h)
    ts_df, period = ts_resample(tsf, cfg, df, fields)

    if period.startswith('Y'):
        # show year only (don't show month and day for Y)
        ts_df.index = ts_df.index.strftime('%Y')
    elif period.startswith('Q'):
        # convert to period for getting quarters
        ts_df.index = ts_df.index.to_period('Q')
        ts_df.index = ts_df.index.strftime('%Y-Q%q')

    algo = 'stl'
    if cfg.get('algo'):
        algo = cfg['algo']

    robust = False
    if cfg.get('robust'):
        robust = True

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    match algo:
        case 'stl':
            # Seasonal-Trend decomposition using LOESS (additive)
            decomp = STL(ts_df[vfield], robust=robust).fit()
            # variance of residuals + seasonality
            resid_seas_var = (decomp.resid + decomp.seasonal).var()
            # seasonal strength
            strength = 1 - (decomp.resid.var() / resid_seas_var)
        case 'add':
            # additive
            decomp = sm_seasonal.seasonal_decompose(ts_df[vfield], model='additive')

        case 'multi':
            # multiplicative
            decomp = sm_seasonal.seasonal_decompose(ts_df[vfield], model='multiplicative')
        case '_':
            return fig

    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[vfield], name=vfield, mode='lines'), 1, 1)
    fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend.values, name='Trend', mode='lines'), 1, 1)
    fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal.values, name='Season', mode='lines'), 2, 1)
    fig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid.values, name='Resid', mode='markers'), 3, 1)
    return fig


"""
plt ts predict chart
# ols(线性普通最小二乘), lowess(局部加权线性回归), rolling(移动平均线), ewm(指数加权移动平均), expanding(扩展窗)
{"pid": "ts", "ts": "time", "field": "dena74",  "period": "MS", "agg": "mean", "algo": "ets", "trend": "add", "season": "add"}
"""
def plt_ts_predict(tsf, cfg, df, fields):
    fig = go.Figure()
    if cfg.get('vf') is None or cfg.get('period') is None:
        return fig

    algo = 'ets'
    if cfg.get('algo'):
        algo = cfg['algo']

    trend = None  # additive or multiplicative
    if cfg.get('trend'):
        trend = cfg['trend']

    season = None
    if cfg.get('season'):
        season = cfg['season']

    future_step = 14
    vf = cfg['vf']
    ts_df, period = ts_resample(tsf, cfg, df, fields)

    match algo:
        case 'ses':
            # Simple Exponential Smoothing(没有趋势和季节性)
            md = SimpleExpSmoothing(ts_df).fit(optimized=True, use_brute=True)
            pred = md.predict(start=len(ts_df)//2, end=len(ts_df) + future_step)
        case 'holt':
            # Holt's linear trend method(有趋势但没有季节性)
            md = Holt(ts_df, initialization_method='estimated', damped_trend=False if trend is None else True).fit(optimized=True)
            pred = md.predict(start=len(ts_df)//2, end=len(ts_df) + future_step)
            # fig.add_trace(go.Scatter(x=md.trend.index, y=md.trend.values + ts_df.mean(), name='Trend', mode='lines'))
        case 'ets':
            # Holt-Winter's additive/multiplicative/damped method(有趋势也有季节性)
            # Cannot compute initial seasonals using heuristic method with less than two full seasonal cycles in the data.
            md = ExponentialSmoothing(ts_df, trend=trend, seasonal=season, damped_trend=False).fit(optimized=True, use_brute=True)
            pred = md.predict(start=len(ts_df)//2, end=len(ts_df) + future_step)
            # fig.add_trace(go.Scatter(x=md.trend.index, y=md.trend.values + ts_df.mean(), name='Trend', mode='lines'))
            # fig.add_trace(go.Scatter(x=md.season.index, y=md.season.values, name='Season', mode='lines'))
        case 'arima':
            # Autoregressive Integrated Moving average (差分整合移动平均自回归模型)
            # sktime.ARIMA requires package 'pmdarima'
            # issue happens when import pmdarima
            # reason: numpy version is high
            # non-seasonal ARIMA for Stationary Time Series. update order(d) if not
            # AR是"自回归", I为差分, MA是"移动平均"
            # ARIMA原理：将非平稳时间序列转化为平稳时间序列然后将因变量仅对它的滞后值以及随机误差项的现值和滞后值进行回归所建立的模型
            # order = (p, d, q), p:自回归项数,偏自相关滞后p阶后变为0, d: 使之成为平稳序列所做的差分次数（阶数）, q: 滑动平均项数,自相关滞后q阶后变为0
            # d=0: MA, d=1: ARMA, d=2: ARIMA
            # seasonal_order = (P, D, Q, s), P:自回归项数, D: 差分阶数, Q:移动平均阶数, s: 季节性周期数
            # D=0: no season, D=1: 季节性自回归, D=2: 季节性ARMA
            order_d = 0 if trend is None else 1 if trend == 'add' else 2
            season_D = 0 if season is None else 1 if season == 'additive' else 2
            md = ARIMA(ts_df.values, order=(0, order_d, 0), trend=None, seasonal_order=(0, season_D, 0, 12))
            md_ft = md.fit()
            tmp = ts_df.head(len(ts_df) // 2)
            y_pred = md_ft.predict(start=len(tmp)-1, end=len(ts_df)+future_step, dynamic=True)
            prange = pd.date_range(tmp.index.max(), periods=len(y_pred), freq=period)
            pred = pd.Series(y_pred.tolist(), index=prange)
        case 'autoarima':
            # 自动差分整合移动平均自回归模型
            # from sktime.datasets import load_airline
            # yyyyy = load_airline()
            temp_df = ts_df.head(len(ts_df) - 7)
            tmp = ts_df.head(len(ts_df)//2)
            # the following issue when import pmdarima
            # reason: numpy version is high
            # numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
            md = AutoARIMA(sp=12, seasonal=True, n_jobs=3, n_fits=3, maxiter=10, trace=True).fit(temp_df)
            prange = pd.date_range(tmp.index.max(), periods=(len(ts_df)//2)+future_step, freq=period)
            pred = md.predict(prange)
        case 'autoets':
            # temp_df.index = temp_df.index.strftime('%Y-%m')
            # autoETS asks Series, not Dataframe.
            # 'M' is available for period, but is deprecated by Datetime
            period_unit = period
            if period and period == 'MS':
                period_unit = 'M'
            if period and period == 'min':
                period_unit = '5min'
            t_series = pd.Series(ts_df.values.T[0], index=ts_df.index.to_period(period_unit))
            # season period estimation
            sp_est = SeasonalityACF()
            sp_est.fit(t_series)
            sp = sp_est.get_fitted_params()["sp"]
            print(f'estimated sp: {sp}')

            if season:
                # season: 'add' or 'mul'
                md = AutoETS(trend=trend, seasonal=season[:3], sp=12).fit(t_series)
            else:
                md = AutoETS(auto=True).fit(t_series)

            # autoETS asks Series, not Dataframe
            # prange = pd.date_range(temp_df.index.max(), periods=future_step+7, freq=period)
            tmp = ts_df.head(len(ts_df) // 2)
            prange = pd.period_range(tmp.index.max(), periods=len(tmp)+future_step, freq=period_unit)
            pred = md.predict(prange)
        case 'prophet':
            # growth: 'linear', 'logistic' or 'flat'
            # seasonality_mode: 'additive' (default) or 'multiplicative'.
            if season is None:
                season = 'additive'
            md = Prophet(seasonality_mode=season, add_country_holidays=dict(country_name='US'), growth='linear')
            md.fit(ts_df)
            tmp = ts_df.head(len(ts_df) // 2)
            prange = pd.date_range(tmp.index.max(), periods=len(tmp)+future_step, freq=period)
            pred = md.predict(fh=prange)
            pred = pred[vf]
        case 'natureprophet':
            # neuralprophet 0.9.0 requires numpy<2.0.0,>=1.25.0, but you have numpy 2.0.2 which is incompatible.
            temp_df = ts_df.head(len(ts_df) - 7)
            '''
            tmp = ts_df.head(len(ts_df) // 2)
            idx_name = temp_df.index.name
            temp_df.reset_index(inplace=True)
            temp_df = temp_df.rename(columns={idx_name: "ds", vf: "y"})
            md = NeuralProphet()
            md.fit(temp_df, freq=period)
            future = md.make_future_dataframe(df=tmp, periods=(len(ts_df)//2)+future_step)
            pred = md.predict(future)
            pred.set_index('ds', inplace=True)
            # trend, yhat
            pred = pred['yhat']
            '''
        case 'tsfm':
            # TimesFMForecaster requires python version to be <3.11,>=3.10
            tmp = ts_df.head(len(ts_df) // 2)
            prange = pd.date_range(tmp.index.max(), periods=len(tmp) + future_step, freq=period)
            md = TimesFMForecaster(context_len=len(ts_df), horizon_len=future_step)
            md.fit(ts_df)
            pred = md.predict(fh=prange)
            pred = pred[vf]
        case 'deepar':
            dataset = PandasDataset(ts_df, target=vf)
            train_set, test_gen = split(dataset, offset=-36)
            test_set = test_gen.generate_instances(prediction_length=12, windows=3)
            md = DeepAREstimator(prediction_length=12, freq=period, trainer_kwargs={"max_epochs": 5}).train(train_set)
            prange = ListDataset([{'start':'1960-01-01', 'target':[400,400,400,400,400,400,400,400,400,400,400,400]}], freq='ME')
            preds = list(md.predict(test_set.input))
            pred = preds[2]

    dt_idx = pred.index
    # aggregated by period (YS,QS,MS,W,D,h,min,s)
    if period.startswith('Y'):
        # show year only (don't show month and day for Y)
        dt_idx = dt_idx.strftime('%Y')
    elif period.startswith('Q'):
        # convert to period for getting quarters
        dt_idx = dt_idx.to_period('Q')
        dt_idx = dt_idx.strftime('%Y%q')
    elif period.startswith('M'):
        dt_idx = dt_idx.strftime('%Y-%m')

    fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[vf], name=vf, mode='lines', connectgaps=True))
    if isinstance(dt_idx, pd.PeriodIndex):
        dt_idx = dt_idx.start_time

    if algo == 'deepar':
            fig.add_trace(go.Scatter(x=dt_idx, y=pred.median, name='Prediction', mode='lines', connectgaps=True))
    else:
        fig.add_trace(go.Scatter(x=dt_idx, y=pred.values, name='Prediction', mode='lines', connectgaps=True))
    return fig


"""
plt ts anomaly detection chart
{"pid": "ts", "ts": "time", "field": "dena74",  "method": "zscore"}
"""
def plt_ts_anomaly(tsf, cfg, df, fields):
    fig = go.Figure()
    if cfg.get('vf') is None:
        return fig

    method = 'quantile'
    if cfg.get('method'):
        method = cfg['method']

    metric = 'euclidean'
    if cfg.get('metric'):
        metric = cfg['metric']

    # contamination: 默认为0.05，即5%的异常值
    # irq_coff: 默认为1.6，即1.6倍IQR
    # sigma_coff: 默认为3，即3倍标准差
    # use same parameter name for all methods
    threshold = None
    if cfg.get('threshold'):
        threshold = cfg['threshold']

    vf = cfg['vf']
    df, period = ts_resample(tsf, cfg, df, fields)
    y_pred = None
    match method:
        case 'quantile':
            # distribution-based
            iqr_coff = threshold if threshold else 1.6

            # get statistics info
            # IQR: InterQuartile range (四分位距)
            stat = df[vf].describe()
            # Inter Quantile Range
            iqr = stat.loc['75%'] - stat.loc['25%']
            th_lower = stat.loc['25%'] - iqr * iqr_coff
            th_upper = stat.loc['75%'] + iqr * iqr_coff
            # 0: normal, 1: outlier
            y_pred = [-1 if v<th_lower else 1 if v>th_upper else 0 for v in df[vf].values]
            th_lower = [th_lower] * len(y_pred)
            th_upper = [th_upper] * len(y_pred)
        case 'zscore':
            # distribution-based
            sigma_coff = threshold if threshold else 3

            # get statistics info
            stat = df[vf].describe()
            # 3 sigma line
            th_lower = (stat.loc['mean'] - stat.loc['std'] * sigma_coff).round(3)
            th_upper = (stat.loc['mean'] + stat.loc['std'] * sigma_coff).round(3)
            y_pred = [-1 if v<th_lower else 1 if v>th_upper else 0 for v in df[vf].values]
            th_lower = [th_lower] * len(y_pred)
            th_upper = [th_upper] * len(y_pred)
        case 'diff':
            diff_coff = threshold if threshold else 2
            diff_df = df[vf].diff(periods=1)
            diff_df.fillna(0)
            diff_std = df[vf].std()

            # 2 std line
            th_lower = (df[vf] - diff_std * diff_coff).round(3)
            th_upper = (df[vf] + diff_std * diff_coff).round(3)

            out1 = [1 if v > diff_std * diff_coff else 0 for v in diff_df.values]
            out2 = [-1 if v < 0-(diff_std * diff_coff) else 0 for v in diff_df.values]
            y_pred = [out1[i] + out2[i] for i in range(len(out1))]
        case 'rolling':
            # Rolling Standard Deviation
            std_coff = threshold if threshold else 2

            # get statistics info
            ts_rolling = df[vf].rolling(window=3, center=True).mean()
            ts_rolling = ts_rolling.ffill().bfill()
            rolling_std = ts_rolling.std()

            # 2 std line
            th_lower = (ts_rolling - rolling_std * std_coff).round(3)
            th_upper = (ts_rolling + rolling_std * std_coff).round(3)

            out1 = [1 if v >0 else 0 for v in (df[vf] - th_upper).values]
            out2 = [-1 if v < 0 else 0 for v in (df[vf] - th_lower).values]
            y_pred = [out1[i]+out2[i] for i in range(len(out1))]
        case 'dbscan':
            # Density-Based Spatial Clustering of Applications with Noise，具有噪声的基于密度的聚类方法(cluster-based)
            #  ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’,
            #  ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
            #  ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’
            # 'minkowski' does not work
            distance = threshold if threshold else 0.5
            clf = DBSCAN(eps=distance, metric=metric)
            clf.fit(df[[vf]])
            y_pred = [1 if i < 0 else 0 for i in clf.labels_]
        case 'cof':
            # Connectivity-Based Outlier Factor (COF, LOF的变种, density-based)
            cont_ratio = threshold if threshold else 0.05
            clf = COF(contamination=cont_ratio, n_neighbors=15)
            clf.fit(df[[vf]])
            y_pred = clf.predict(df[[vf]])
        case 'vae':
            # AutoEncoder(自编码器, unsupervised, neural network)
            cont_ratio = threshold if threshold else 0.05
            clf = vae.VAE(epoch_num=50, batch_size=32, contamination=cont_ratio)
            clf.fit(df[[vf]])
            # 1: outlier
            y_pred = clf.labels_
        case 'dif':
            # Deep Isolation Forest
            cont_ratio = threshold if threshold else 0.05
            clf = dif.DIF(batch_size=32, contamination=cont_ratio, n_estimators=10)
            clf.fit(df[[vf]])
            y_pred = clf.labels_
        case 'ecod':
            cont_ratio = threshold if threshold else 0.05
            clf = ecod.ECOD(contamination=cont_ratio)
            clf.fit(df[[vf]])
            y_pred = clf.labels_
        case 'dsvdd':
            cont_ratio = threshold if threshold else 0.05
            clf = deep_svdd.DeepSVDD(n_features=1, epochs=100, contamination=cont_ratio)
            clf.fit(df[[vf]])
            y_pred = clf.labels_
        case 'ae1svm':
            cont_ratio = threshold if threshold else 0.05
            clf = ae1svm.AE1SVM(contamination=cont_ratio)
            clf.fit(df[[vf]])
            y_pred = clf.labels_
        case 'ruptures':
            # segment detection
            penalty = threshold if threshold else 5
            # model: l1, l2, rbf
            # Linearly penalized segmentation (Pelt)
            algo = rpt.Pelt(model="rbf").fit(df[vf].values)
            bounds = algo.predict(pen=penalty)

            # Dynamic programming (Dynp)
            # algo = rpt.Dynp(model="l2").fit(df[vf].values)
            # bounds = algo.predict(n_bkps=penalty)

            # Binary segmentation (Binseg)
            # algo = rpt.Binseg(model='normal', min_size=7).fit(df[vf].values)
            # bounds = algo.predict(n_bkps=3)

            # Bottom-up segmentation (BottomUp)
            # algo = rpt.BottomUp(model='ar').fit(df[vf].values)
            # bounds = algo.predict(n_bkps=3)

            # Window-Based
            # algo = rpt.Window(model="l2", width=28).fit(df[vf].values5
            # bounds = algo.predict(n_bkps=3)

            # Kernel Change Point Detection (KernelCPD)
            # algo = rpt.KernelCPD(kernel='rbf', min_size=8).fit(df[vf].values)
            # bounds = algo.predict(n_bkps=3)

            v_min = math.floor(df[vf].min())
            v_max = math.ceil(df[vf].max())
            y_pred = []
        case '_':
            return fig

    # two subplots
    fig = make_subplots(rows=2, cols=1, row_heights=[600, 200], vertical_spacing=0.07)
    # original data line
    fig.add_trace(go.Scatter(x=df.index, y=df[vf], name=vf, hovertemplate='%{y}<extra></extra>'), 1, 1)

    # outlier markers
    if len(y_pred) > 0:
        abs_y = [abs(x) for x in y_pred]
        outlier_df = pd.DataFrame(df[vf] * abs_y)
        outlier_df = outlier_df[outlier_df[vf] > 0]
        fig.add_trace(go.Scatter(x=outlier_df.index, y=outlier_df[vf], name='outlier', line=dict(color="#ff0000"),
                                 mode='markers', hovertemplate='%{y}<extra></extra>'), 1, 1)

        # outliers with binary index (0, 1 or -1)
        fig.add_trace(go.Scatter(x=df.index, y=y_pred, name='binary index', hovertemplate='%{y}<extra></extra>'), 2, 1)
        fig.update_layout(title=f'total outliers: {len(outlier_df)}', hovermode='x')

    if method in ['quantile', 'zscore', 'diff', 'rolling']:
        # add tolerance area
        fig.add_trace(go.Scatter(x=df.index, y=th_lower, name='lower', showlegend=False,
                                 line=dict(color='rgba(255,228,181,0.2)', width=0, dash='dot')), 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=th_upper, name='upper', showlegend=False, fill='tonexty',
                                 line=dict(color='rgba(255,228,181,0.2)', width=0, dash='dot')), 1, 1)

    if method == 'ruptures':
        if len(bounds) > 1:
            df['tmp_idx'] = range(len(df))
            for b in bounds:
                # bounds has the last index of df always
                # ex: bounds==[len(df)] when df has only ONE segment
                if b != bounds[-1]:
                    fig.add_shape(type="line", x0=df[df['tmp_idx']==b].index[0], y0=v_min, x1=df[df['tmp_idx']==b].index[0],
                              y1=v_max, line=dict(color="gray", dash="dot"), row=1, col=1)
            fig.update_layout(title=f'total segments: {len(bounds)}', hovermode='x')
            reshaped_df = ts_segment_reshaping(tsf, cfg, df, bounds)
            if reshaped_df is not None:
                fig.add_trace(
                    go.Scatter(x=reshaped_df.index, y=reshaped_df[vf], name='Reshape', hovertemplate='%{y}<extra></extra>'), 2, 1)
    return fig


"""
plt ts active noice reduction chart
{"pid": "ts", "ts": "time", "field": "dena74",  "method": "zscore"}
"""
def plt_ts_anc(tsf, cfg, df, fields):
    fig = go.Figure()
    if cfg.get('vf') is None:
        return fig

    method = 'quantile'
    if cfg.get('method'):
        method = cfg['method']

    metric = 'euclidean'
    if cfg.get('metric'):
        metric = cfg['metric']

    # contamination: 默认为0.05，即5%的异常值
    # irq_coff: 默认为1.6，即1.6倍IQR
    # sigma_coff: 默认为3，即3倍标准差
    # use same parameter name for all methods
    threshold = None
    if cfg.get('threshold'):
        threshold = cfg['threshold']

    vf = cfg['vf']
    df, period = ts_resample(tsf, cfg, df, fields)
    y_pred = None
    match method:
        case 'fft':
            # default value is 1k Hz
            cutoff_freq = threshold*1000 if threshold else 1000
            # fft
            fft_vals = np.fft.rfft(df[vf].values)

            # get the frequencies
            fft_freqs = np.fft.rfftfreq(len(df[vf]), 1/1000)
            # apply the filter mask to the FFT values
            fft_clean = fft_vals * (fft_freqs <= cutoff_freq)
            # reverse the FFT to get the filtered signal
            df['anc'] = np.fft.irfft(fft_clean, n=len(df[vf]))

            fft_amps = np.abs(fft_vals) / len(df[vf])
            fft_amps = np.round(fft_amps, 4)
            fft_freqs = np.round(fft_freqs/1000, 4)

            # PSD(Power spectral density, 功率谱密度)
            psd_freq, psd_power = periodogram(df[vf])
            psd_freq = np.round(psd_freq, 4)
            psd_power = np.round(psd_power, 4)

            psd_df = pd.DataFrame({'freq': psd_freq, 'power': psd_power})
            psd_df = psd_df.sort_values(by='power', ascending=False)
            # get top 5 by power
            top_power = psd_df.head(5)
            top_power = top_power[top_power['power'] > 0]

            # sort by index
            psd_df.sort_index(inplace=True)
            top_power.sort_index(inplace=True)
        case '_':
            return fig

    # two subplots
    fig = make_subplots(rows=2, cols=1, row_heights=[500, 300], vertical_spacing=0.05)
    # original data line
    fig.add_trace(go.Scatter(x=df.index, y=df[vf], name=vf, hovertemplate='%{y}<extra></extra>'), 1, 1)
    # denoised data line
    fig.add_trace(go.Scatter(x=df.index, y=df['anc'], name='Denoised', hovertemplate='%{y}<extra></extra>'), 1, 1)
    # fft line
    # fig.add_trace(go.Scatter(x=fft_freqs, y=fft_amps, name='FFT', hovertemplate='%{y}<extra></extra>'), 3, 1)
    # psd line
    fig.add_trace(go.Scatter(x=psd_df['freq'], y=psd_df['power'], name='PSD', hovertemplate='%{y}<extra></extra>'), 2, 1)
    # top 5 freq markers
    fig.add_trace(go.Scatter(x=top_power['freq'], y=top_power['power'], text=top_power['freq'], name='Freq',
                             textposition='top right', showlegend=False, mode='markers+text', hovertemplate='%{y}<extra></extra>'), 2, 1)
    fig.update_xaxes(row=2, col=1, title='Freq')
    fig.update_layout(hovermode='x')
    return fig



def ts_resample(tsf: str, cfg: any, df: pd.DataFrame, fields: list):
    period = cfg.get('period', None)
    if period is None:
        return df, period

    # aggregated by period (YS,QS,MS,W,D,h,min,s)
    # agg: sum, mean, median, min, max, std, count
    agg = cfg['agg'] if cfg.get('agg') else 'mean'
    for field in fields:
        # find config of ts field to confirm period
        if field.get('name') == tsf and field.get('gap'):
            period_min = 1
            if field.get('gap'):
                period_min = int(field['gap'])
            resample_min = 1
            if field.get('resample'):
                resample_min = int(field['resample'])
            final_period = max(period_min, resample_min)
            if final_period < 60 and (period == 'min' or period == 's'):
                # update period to min period of this time series
                period = f'{final_period}T'
                break
            elif final_period > 60 and final_period < 1440 and (period == 'h' or period == 'min' or period == 's'):
                # update period to min period of this time series
                period_h = math.ceil(final_period / 60)
                period = f'{period_h}H'
                break

    resampled_df = None
    if cfg.get('cat'):
        cat_field = cfg['cat']
        # get page info
        page = cfg['page'] if cfg.get('page') else 0
        # unique categories
        cat_values = df[cat_field].unique()
        # sort categories
        cat_values.sort()
        # get current page. 50 categories
        cat_50 = cat_values[page * 50:(page + 1) * 50]
        # filter data by cat
        cat50_df = df[df[cat_field].isin(cat_50)]

        # resample each category
        for cat in cat_50:
            # filter data by cat
            cat_df = cat50_df[cat50_df[cat_field] == cat]
            # drop cat field
            cat_df.drop(columns=[cat_field], inplace=True)
            # resample
            cat_df = cat_df.resample(period).agg(agg)
            # get cat field back
            cat_df[cat_field] = cat
            if resampled_df is None:
                resampled_df = cat_df
            else:
                resampled_df = pd.concat([resampled_df, cat_df])
    else:
        resampled_df = df.resample(period).agg(agg)

    # value field
    num_fields = [field for field in fields if field['attr'] in ['conti', 'disc']]
    # handle missing value
    for field in num_fields:
        vf = field['name']
        if vf in resampled_df.columns and resampled_df[vf].isnull().any() and field.get('miss'):
            match field['miss']:
                case 'mean':
                    resampled_df[vf] = resampled_df[vf].fillna(resampled_df[vf].mean())
                case 'median':
                    resampled_df[vf] = resampled_df[vf].fillna(resampled_df[vf].median())
                case 'mode':
                    resampled_df[vf] = resampled_df[vf].fillna(resampled_df[vf].mode())
                case 'prev':
                    resampled_df[vf] = resampled_df[vf].ffill()
                case 'next':
                    resampled_df[vf] = resampled_df[vf].bfill()
                case 'zero':
                    resampled_df[vf] = resampled_df[vf].fillna(value=0)
    return resampled_df, period

"""
plt ts anomaly detection chart
{"pid": "ts", "ts": "time", "field": "dena74",  "method": "zscore"}
"""
def ts_segment_reshaping(tsf: str, cfg: dict, df: pd.DataFrame, segments: list):
    if (len(df) < 7) or (len(segments) < 2):
        return df

    vf = cfg.get('vf')
    reshaped_df = None
    # segments always has a value at least
    for idx, seg in enumerate(segments):
        stone = None
        if idx == 0:
            seg_df = df.head(seg)
            # drop 2 boundary points for mean and std
            stone = seg_df.tail(2)
            seg_df.drop(stone.index, inplace=True)
        elif idx == len(segments) - 1:
            seg_df = df[segments[idx-1]:seg]
            # drop 2 boundary points for mean and std
            stone = seg_df.head(2)
            seg_df.drop(stone.index, inplace=True)
        else:
            seg_df = df[segments[idx - 1]:seg]
            # drop 4 boundary points for mean and std
            stone = seg_df.head(2)
            stone = pd.concat([stone, seg_df.tail(2)])
            seg_df.drop(stone.index, inplace=True)

        if seg_df[vf].std() < 1:
            # it is outlier if it is out of 2 sigma
            mean = seg_df[vf].mean()
            th_lower = mean - seg_df[vf].std() * 3
            th_lower = min(th_lower, mean - abs(mean * 0.1))
            th_upper = mean + seg_df[vf].std() * 3
            th_upper = max(th_upper, mean + abs(mean * 0.1))
            # filter out outliers
            # replace segment with current mean which exclude outlier
            seg_df[(seg_df[vf]<th_upper) | (seg_df[vf]>th_lower)] = seg_df[vf].mean()

        # get boundary points back
        seg_df = pd.concat([seg_df, stone])
        # sort date index
        seg_df.sort_index(inplace=True)
        if reshaped_df is None:
            reshaped_df = seg_df
        else:
            reshaped_df = pd.concat([reshaped_df, seg_df])

    return reshaped_df