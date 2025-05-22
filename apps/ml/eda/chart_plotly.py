import math
import operator
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import torch
import umap
from gluonts.dataset.common import ListDataset
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator
from minisom import MiniSom
from pyod.models import vae, dif, ecod, ae1svm, deep_svdd
from pyod.models.knn import KNN
from sklearn import manifold
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, KernelPCA, FastICA, TruncatedSVD
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, silhouette_score
from sktime.clustering.k_means import TimeSeriesKMeans
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
from scipy.stats import chi2

"""
build chart of statistics
"""
def plt_stat_chart(kind, config, df, fields):
    # get valid fields (some columns are removed when data is loaded if they have all null value)
    df_cols = df.columns.tolist()
    valid_f = [field for field in fields if ('omit' not in field) and (field['name'] in df_cols)]

    # generate chart based on kind
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
        case 'volume':
            fig = plt_dist_volume(config, df, fields)
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
        case 'cov':
            # Covariance Matrix
            fig = plt_corr_cov(config, df, fields)
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
build chart of clustering
"""
def plt_clustering_chart(kind, config, df, fields):
    # generate chart based on kind
    match kind:

        case 'kmeans':
            # single scatter with margins
            fig = plt_cluster_kmeans(config, df, fields)
        case 'dbscan':
            # scatter matrix
            fig = plt_cluster_kmeans(config, df, fields)
        case _:
            # do nothing
            fig = None

    return fig

"""
dim reduction chart
"""
def plt_reduction_chart(kind, config, df, fields):
    cat = None
    t_values = None

    # target field
    target_field = [field['name'] for field in fields if 'target' in field]
    if target_field is not None and len(target_field) > 0:
        f_df = df.drop(columns=target_field)
        for ele in fields:
            # find unique values of category field
            if ele['name'] == target_field[0] and ele['attr'] == 'cat':
                t_values = ele.get('values')
                cat = ele['name']
                break
    else:
        f_df = df

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

    metric = 'euclidean'
    if config.get('metric'):
        metric = config['metric']

    title = None
    labels = {f'd{i}': kind.upper() + f"d{i}" for i in range(dim)}

    # only numerical columns
    f_df = f_df.select_dtypes(include='number')
    fig = go.Figure()
    match kind:
        case 'pca':
            if config.get('kernel') and config['kernel'] != 'linear':
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
                    f'd{i}': f"d{i} ({var:.1f}%)"
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
            pca = manifold.TSNE(n_components=dim, perplexity=perplex, metric=metric, init='pca')
            data = pca.fit_transform(f_df)
        case 'umap':
            # Uniform Manifold Approximation and Projection(等距 manifold 近似和投影算法)
            # 解决MDS算法在非线性结构数据集上的弊端,数据在向低维空间映射之后能够保持流形不变
            # 常用于手写数字等数据的降维
            # 无监督，非线性
            pca = umap.UMAP(n_components=dim, metric=metric, n_neighbors=neighbor)
            data = pca.fit_transform(f_df)
        case 'isomap':
            # isometric mapping (等度量映射算法)
            # 解决MDS算法在非线性结构数据集上的弊端,数据在向低维空间映射之后能够保持流形不变
            # 常用于手写数字等数据的降维
            # 无监督，非线性
            pca = manifold.Isomap(n_components=dim, metric=metric, n_neighbors=neighbor)
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
                f'd{i}': f"d{i} ({var:.1f}%)"
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

    # get data center
    center = np.mean(data, axis=0)

    # add 2 or 3 dim data to original dataset
    dim_df = pd.DataFrame(data, columns=[f'd{i}' for i in range(dim)])
    df = pd.concat([df, dim_df], axis=1)
    vis_cols = dim_df.columns.tolist()

    # tooltip title
    df.reset_index(inplace=True)
    hover_name = config.get('label', 'index')
    num_fields = [field['name'] for field in fields if field['attr'] in ('conti', 'disc')]
    hover_data = num_fields

    if t_values and df[cat].dtype == 'category':
        # replace codes to category names
        df[cat] = df[cat].cat.rename_categories(t_values)

    if dim == 1:
        fig = px.scatter(x=df.index, y=df['d0'], color=None if cat is None else dim_df[cat],
                         labels=labels, title=title, hover_name=hover_name, hover_data=num_fields)
        # add center point
        fig.add_trace(go.Scatter(x=[center[0]], name='Center', mode='markers',
                                 marker=dict(color='black', size=10, symbol='x')))

    elif dim == 2:
        fig = px.scatter(df, x='d0', y='d1', color=None if cat is None else dim_df[cat],
                         labels=labels, title=title, hover_name=hover_name)
        # add center point
        fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], name='Center', mode='markers',
                                 marker=dict(color='black', size=10, symbol='x')))

        if config.get('threshold'):
            # 计算协方差矩阵及其逆矩阵
            cov_matrix = np.cov(data, rowvar=False)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            # 计算每个点到数据中心的马氏距离
            diff = data - center
            mahal_dist = np.sqrt(np.einsum('ij, ij -> i', diff.dot(inv_cov_matrix), diff))
            # 根据卡方分布选择阈值，假设我们想要保留97.5%的样本
            threshold = chi2.ppf(config['threshold'], df=data.shape[1])
            # 获取特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # 计算半长轴和半短轴
            semi_major_axis = np.sqrt(threshold * eigenvalues[1])
            semi_minor_axis = np.sqrt(threshold * eigenvalues[0])
            # 计算椭圆的角度
            angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
            # 生成椭圆上的点
            theta = np.linspace(0, 2 * np.pi, 100)
            ellipse_x = semi_major_axis * np.cos(theta)
            ellipse_y = semi_minor_axis * np.sin(theta)
            # 旋转并平移椭圆到正确的位置
            R = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                          [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
            rotated_ellipse = np.dot(np.array([ellipse_x, ellipse_y]).T, R)
            final_ellipse_x, final_ellipse_y = rotated_ellipse[:, 0] + center[0], rotated_ellipse[:, 1] + center[1]
            # 添加椭圆
            fig.add_trace(
                go.Scatter(x=final_ellipse_x, y=final_ellipse_y, mode='lines', name='MD boundary', line=dict(color='red')))

    elif dim == 3:
        fig = px.scatter_3d(df, x='d0', y='d1', z='d2', size=[5]*len(dim_df), size_max=10, labels=labels, title=title,
                            color=None if cat is None else dim_df[cat], hover_name=hover_name)
        # add center point
        fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], z=[center[2]], name='Center', mode='markers',
                                 marker=dict(color='black', size=10, symbol='x')))
    else:
        fig = go.Figure(go.Table(header=dict(values=[kind + f'd{i}' for i in range(dim)]),
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
def plt_ts_chart(kind, config, df, fields, transform):
    ts_field = None
    date_fields = [field['name'] for field in fields if field['attr'] == 'date']
    cat_fields = [field['name'] for field in fields if field['attr'] == 'cat']
    value_fields = [field['name'] for field in fields if field['attr'] in ('conti', 'disc')]
    ts_fields = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()

    if config.get('tf'):
        # selected time field
        ts_field = config['tf']
    elif len(date_fields):
        # the first defined date field
        ts_field = date_fields[0]
    elif len(ts_fields):
        # the first date field which detected by Pandas
        # should not come here
        ts_field = ts_fields[0]
    else:
        # no ts field
        return None

    # config['vf'] may be string or list
    vf = config.get('vf')
    if config.get('vf') and isinstance(config.get('vf'), str):
        # convert to list for the following processing
        vf = [config.get('vf')]

    # config['cat'] may be string or list
    cf = config.get('cat')
    if cf and isinstance(cf, list):
        if len(cf) > 1:
            # combine cat fields into a new column
            new_cf = '_'.join(cf)
            df[new_cf] = df[cf].astype(str).agg('_'.join, axis=1)
            # update cf with new cat field and convert it to a string from list
            cf = config['cat'] = new_cf
        else:
            # convert cf to a string from list
            cf = config['cat'] = cf[0]

    # filter selected category fields and value fields
    final_fields = fields
    group_fields = [ts_field]
    # vf is list but cf is string
    if vf:
        if cf:
            # only selected ts field, cat field and value field
            df = df[[ts_field, cf]+vf]
            final_fields = [field for field in fields if field['name'] in [ts_field, cf]+vf]
            group_fields.append(cf)
        else:
            # only selected ts field and value field (omit cat fields)
            df = df[[ts_field]+vf]
            final_fields = [field for field in fields if field['name'] in [ts_field]+vf]
    else:
        if cf:
            # drop all other category fields
            if cf in cat_fields:
                cat_fields.remove(cf)
            df = df.drop(columns=cat_fields, errors='ignore')
            # ts field, selected cat field and all value fields
            final_fields = [field for field in fields if field['name'] in [ts_field, cf]+value_fields]
            group_fields.append(cf)
        elif len(cat_fields):
            # remove all cat fields when no category field is selected
            df = df.drop(columns=cat_fields)
            # ts field and all value fields
            final_fields = [field for field in fields if field['name'] in [ts_field] + value_fields]
        # update vf to include all value fields when no value field is selected
        value_fields.sort()
        vf = value_fields

    # vf is list but cf is string
    if cf:
        # sort by ts field and cat field
        df.sort_values(by=[ts_field, cf], key=lambda x: x.astype(str) if x.name == cf else x, inplace=True)
    # set ts field as index
    df.set_index(ts_field, inplace=True)

    match kind:
        case 'series':
            config['vf'] = vf
            # don't fill null values for time series gap displaying/detection
            df, config['period'] = ts_resample(df, config, fields, transform, False)
            # filter ts data
            df, config = ts_filter(df, config)
            # time series lines
            fig = plt_ts_series(df, config, final_fields)
        case 'tsfreq':
            # quarterly, monthly, weekly and daily
            fig = plt_ts_freq(df, config, final_fields)
        case 'trend':
            # don't fill null values for time series gap displaying/detection
            df, config['period'] = ts_resample(df, config, fields, transform, False)
            # Trending line
            fig = plt_ts_trend(df, config, final_fields)
        case 'diff':
            config['vf'] = vf
            df, config['period'] = ts_resample(df, config, fields, transform, False)
            # difference curve
            fig = plt_ts_diff(df, config, final_fields)
        case 'compare':
            # Comparison
            fig = plt_ts_compare(df, config, final_fields)
        case 'mavg':
            df, config['period'] = ts_resample(df, config, fields, transform, False)
            # Moving Average
            fig = plt_ts_mavg(df, config, final_fields)
        case 'autocorr':
            df, config['period'] = ts_resample(df, config, fields, transform, False)
            # ACF/PACF
            fig = plt_ts_acf(df, config, final_fields)
        case 'tsdist':
            # mean+std, box and violin
            fig = plt_ts_distribution(df, config, final_fields)
        case 'cycle':
            df, config['period'] = ts_resample(df, config, fields, transform, False)
            # Periodicity detection
            fig = plt_ts_cycle(df, config, final_fields)
        case 'decomp':
            df, config['period'] = ts_resample(df, config, fields, transform, False)
            # decomposition(trend + season)
            fig = plt_ts_decomp(df, config, final_fields)
        case 'predict':
            # fill null values with specific config or global transform
            # config['miss'] has higher priority than transform['miss']
            df, config['period'] = ts_resample(df, config, fields, transform, True)
            # prediction
            fig = plt_ts_predict(df, config, final_fields)
        case 'anomaly':
            if config.get('method') == 'gaps':
                df, config['period'] = ts_resample(df, config, fields, transform, False)
            else:
                df, config['period'] = ts_resample(df, config, fields, transform, True)
            # anomaly detection
            fig = plt_ts_anomaly(df, config, final_fields)
        case 'similarity':
            config['vf'] = vf
            df, config['period'] = ts_resample(df, config, fields, transform, True)
            # similarity detection
            fig = plt_ts_similarity(df, config, final_fields)
        case 'anc':
            df, config['period'] = ts_resample(df, config, fields, transform, True)
            # active noice reduction
            fig = plt_ts_anc(df, config, final_fields)
        case _:
            fig = None

    return fig




"""
filter time series based on config 'filter
"""
def ts_filter(df, cfg):
    if cfg.get('filter') is None:
        return df, cfg

    ops = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne
    }

    filter_str = cfg['filter']
    # get operator
    if '>=' in filter_str:
        op_str = '>='
    elif '>' in filter_str:
        op_str = '>'
    elif '<=' in filter_str:
        op_str = '<='
    elif '<' in filter_str:
        op_str = '<'
    elif '==' in filter_str:
        op_str = '=='
    elif '!=' in filter_str:
        op_str = '!='
    else:
        op_str = None

    # get threshold value
    threshold = None
    if op_str:
        split_filter = filter_str.split(op_str)
        threshold = split_filter[-1]
        threshold = float(threshold)

    if threshold:
        op_func = ops.get(op_str)
        if cfg.get('cat'):
            # df has category field
            cf = cfg['cat']
            keep_cat = []
            for cat in df[cf].unique():
                cat_df = df[df[cf] == cat]
                if op_func(cat_df[cfg['vf']], threshold).any().any():
                    keep_cat.append(cat)
            df = df[df[cf].isin(keep_cat)]
        else:
            # df has value fields only
            df = df.loc[:, op_func(df, threshold).any(axis=0)]
            # update value fields
            cfg['vf'] = df.columns.tolist()
            cfg['vf'].sort()
    return df, cfg

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
    df_cols = df.columns.tolist()
    valid_f = [field for field in fields if ('omit' not in field) and (field['name'] in df_cols)]
    omitted_f = [field for field in fields if 'omit' in field]
    attr_names = ['Continuous', 'Discrete', 'Category', 'Datetime', 'Coordinate', 'Invalid']
    attr_values = [len([field for field in valid_f if field['attr'] == 'conti']),
                   len([field for field in valid_f if field['attr'] == 'disc']),
                   len([field for field in valid_f if field['attr'] == 'cat']),
                   len([field for field in valid_f if field['attr'] == 'date']),
                   len([field for field in valid_f if field['attr'] == 'coord']),
                   len(fields) - len(valid_f) - len(omitted_f)]

    # rows with missing value
    miss_r = df.isnull().sum(axis=1)
    miss_r = miss_r[miss_r.values > 0]
    # columns with missing value
    miss_df = df.isnull().sum().to_frame()
    miss_df.reset_index(inplace=True, names=['name'])
    miss_df.rename(columns={0: 'miss'}, inplace=True)

    num_fields = [field['name'] for field in valid_f if field['attr'] in ('conti', 'disc')]

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

    title = 'Box plot (mean)'
    if boxmean == 'sd' and outlier == 'outliers':
        title = 'Box plot (mean, std dev and outlier)'
    elif boxmean == 'sd':
        title = 'Box plot (mean and std dev)'
    elif outlier == 'outliers':
        title = 'Box plot (mean and outlier)'

    # category field
    cf = None
    if cfg.get('cf'):
        cf = cfg['cf']
        title += f', Category field: {cf}'

    # quantilemethod: linear, inclusive, exclusive
    fig = go.Figure()
    if cf is None or cf == '':
        for name in sorted_name:
            fig.add_trace(go.Box(y=df[name], name=name, boxmean=boxmean, boxpoints=outlier))
    else:
        c_values = df[cf]
        '''
        for f in fields:
            if f['name'] == cf and f.get('values'):
                c_values = df[cf].map(dict(enumerate(f['values'])))
                break
        '''
        for name in sorted_name:
            fig.add_trace(go.Box(x=c_values, y=df[name], name=name, boxmean=boxmean, boxpoints=outlier))
        fig.update_layout(boxmode='group')

    fig.update_layout(title=title)
    return fig

"""
Violin plot for numeric fields
"""
def plt_stat_violin(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                 [field['name'] for field in fields if field['attr'] == 'disc']
    sorted_name = df[[n for n in num_fields]].mean(numeric_only=True).sort_values().index
    title = 'Violin plot (mean and std dev)'

    box = False
    if cfg.get('box'):
        box = cfg['box']

    outlier = False
    if cfg.get('outlier'):
        outlier = 'outliers'
        title = 'Violin plot (mean, std dev and outlier)'

    # category field
    cf = None
    if cfg.get('cf'):
        cf = cfg['cf']
        title += f', Category field: {cf}'


    # quantilemethod: linear, inclusive, exclusive
    fig = go.Figure()
    if cf is None or cf == '':
        for name in sorted_name:
            fig.add_trace(go.Violin(y=df[name], name=name, points=outlier, box_visible=box, meanline_visible=True))
    else:
        c_values = df[cf]
        '''
        for f in fields:
            if f['name'] == cf and f.get('values'):
                c_values = df[cf].map(dict(enumerate(f['values'])))
                break
        '''
        for name in sorted_name:
            fig.add_trace(go.Violin(x=c_values, y=df[name], name=name, points=outlier, box_visible=box, meanline_visible=True))
        fig.update_layout(violinmode='group')

    fig.update_layout(title=title)
    return fig

"""
Omega squared and eta squared
"""
def calculate_effect_size(md, anova_table, type='omega_sq'):
    if type == 'eta_sq':
        anova_table['effect'] = anova_table['sum_sq'] / sum(anova_table['sum_sq'])
    else:
        # 手动计算总平方和 SSTotal = SSE（残差） + SSB（因子）
        SSTotal = np.var(md.model.endog, ddof=0) * len(md.model.endog)
        SSB = anova_table['sum_sq'].sum()  # 所有因子的平方和之和
        SSE = SSTotal - SSB  # 残差平方和 = 总平方和 - 组间平方和

        # 自由度
        DFB_total = anova_table['df'].sum()
        DFT = len(md.model.endog) - 1
        DFE = DFT - DFB_total  # 残差自由度

        # 计算 omega-squared
        anova_table['effect'] = (anova_table['sum_sq'] - (anova_table['df'] * SSE / DFE)) / (SSTotal + SSE)

    bins = [0, 0.01, 0.05, 0.1, 1]
    labels = ['high', 'medium', 'low', 'no']
    anova_table['Significant'] = pd.cut(anova_table['PR(>F)'], bins=bins, labels=labels, right=False)
    return anova_table

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
    num_fields = [field['name'] for field in fields if field['attr'] in ('conti', 'disc')]
    cat_fields = [field['name'] for field in fields if field['attr'] == 'cat']
    fig = go.Figure()
    # calculate all numeric fields based on selected category field (1 cat: one-way, 2cats: two-way)
    # calculate all discrete fields based on selected continuous field
    if cfg.get('field') is None:
        return fig

    sel_field = cfg['field']
    sel_attr = None
    sel_values = None
    title = ''
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

        title = f'One-way ANOVA, Category field: {sel_field}({len(cat_names)})'
        num_col = 3
        if len(num_fields) < 3:
            num_col = len(num_fields)

        num_row = math.ceil(len(num_fields) / num_col)
        fig = make_subplots(rows=num_row, cols=num_col,
                            subplot_titles=['{}'.format(n) for n in num_fields],
                            row_heights=[400 for i in range(num_row)],
                            horizontal_spacing=0.05, vertical_spacing=0.2 / num_row)
        for idx, name in enumerate(num_fields):
            # mean and std of every category group
            mean_std = df.groupby(by=sel_field, observed=True)[name].agg(['mean', 'std']).reset_index()
            if cfg.get('order'):
                # sort by 'mean' or 'std'
                mean_std.sort_values(by=cfg.get('order'), ascending=False, inplace=True)

            # the data of every category group
            cat_data = [df[df[sel_field] == c][name] for c in cat_list]
            # get p value then insert to mean_std
            # F = MSB/MSE(组间方差与组内方差的比率，对比F分布统计表，如果大于临界值，说明总体均值间存在差异)
            # pValue<0.05，说明分类数据间存在显著差异
            # 当均值差异较大且标准差较小时，p值通常较小，这表明统计显著性较强。
            # 相反，当均值差异较小或者标准差较大时，p值通常较大，这表明统计显著性较弱。
            # p值可以看作是一种综合考虑了均值差异和标准差差异的指标
            f_v, p_v = stats.f_oneway(*cat_data)
            # mean_std.insert(2, 'pvalue', np.round(p_v, 3))
            if cfg.get('style') == 'line':
                fig.add_trace(go.Scatter(x=mean_std[sel_field], y=mean_std['mean'].round(3), name='',
                                         error_y=dict(type='data', array=mean_std['std'].round(3))),
                              (idx // num_col) + 1, (idx % num_col) + 1)
            else:
                # mean-std chart by categories for selected numeric field
                fig.add_trace(go.Bar(x=mean_std[sel_field], y=mean_std['mean'].round(3), name='',
                                     error_y=dict(type='data', array=mean_std['std'].round(3))),
                              (idx // num_col) + 1, (idx % num_col) + 1)
            # update subplot title to add p-value
            if p_v < 0.01:
                fig.layout.annotations[idx].text = f'{name} (p={np.round(p_v*100, 1)}%, high)'
            elif p_v < 0.05:
                fig.layout.annotations[idx].text = f'{name} (p={np.round(p_v*100, 1)}%, medium)'
            elif p_v < 0.1:
                fig.layout.annotations[idx].text = f'{name} (p={np.round(p_v*100, 1)}%, low)'
            else:
                fig.layout.annotations[idx].text = f'{name} (p={np.round(p_v*100, 1)}%, no)'
        fig.update_layout(title=title, showlegend=False, yaxis_title='Mean-Std Dev')
        fig.update_xaxes(type='category')
    elif sel_attr == 'conti' or sel_attr == 'disc':
        # multiple-factor anova (包括Two-way ANOVA)
        # 分析两个及以上分类特征对一个数值特征的影响程度
        # 模型的公式为“y ~ A + B + C + A*B + A*C + B*C + A*B*C”，
        # y是因变量，A、B和C是自变量。A*B、A*C和B*C是自变量的交互作用。
        # "y ~ A + B + C" 表示不考虑交互作用，而只考虑A、B和C三个自变量对y的影响。
        title = f'Multiple-factor ANOVA, Value field: {sel_field}'
        formula = sel_field + '~'
        for name in cat_fields:
            formula += '+' + name
        ana = smf.ols(formula, data=df).fit()
        fp_df = sm.stats.anova_lm(ana, type=2).dropna(how='any')
        # PR(>F) is p-value, <5%: 显著相关
        # F值越大，p值越小，p值越小，越显著
        # convert p-value to -log10() for displaying
        # -log10(p-value) 越大越显著
        # 1e-308 is used to avoid division by zero (inf->308)
        fp_df['neg_log10'] = -np.log10(fp_df['PR(>F)'].replace(0, 1e-308))
        effect_type = cfg.get('effect', 'omega_sq')

        fp_df = calculate_effect_size(ana, fp_df, effect_type)
        effect_threshold1 = 0.06
        effect_threshold2 = 0.14
        p_threshold1 = -np.log10(0.01)
        p_threshold2 = -np.log10(0.05)
        p_threshold3 = -np.log10(0.1)
        if cfg.get('style') == 'scatter':
            color_map = {'high': 'red', 'medium': 'orange', 'low': 'green', 'no': 'gray'}
            fig = px.scatter(fp_df, x='effect', y='neg_log10', color='Significant', text=fp_df.index,
                             color_discrete_map=color_map,
                             title=title, labels={'effect': f'Effect({effect_type})', 'neg_log10': '-Log10(p)'})
            fig.update_traces(marker_size=10, textposition="bottom center")
            fig.update_layout(
                shapes=[
                    # reference line for Effect size
                    dict(type="line", x0=effect_threshold1, x1=effect_threshold1, y0=fp_df['neg_log10'].min(),
                         y1=fp_df['neg_log10'].max(),
                         line=dict(color="green", width=2, dash="dash")),
                    dict(type="line", x0=effect_threshold2, x1=effect_threshold2, y0=fp_df['neg_log10'].min(),
                         y1=fp_df['neg_log10'].max(),
                         line=dict(color="orange", width=2, dash="dash")),
                    # reference line for p value
                    dict(type="line", x0=fp_df['effect'].min(), x1=fp_df['effect'].max(), y0=p_threshold1,
                         y1=p_threshold1,
                         line=dict(color="orange", width=2, dash="dash")),
                    dict(type="line", x0=fp_df['effect'].min(), x1=fp_df['effect'].max(), y0=p_threshold2,
                         y1=p_threshold2,
                         line=dict(color="green", width=2, dash="dash")),
                    dict(type="line", x0=fp_df['effect'].min(), x1=fp_df['effect'].max(), y0=p_threshold3,
                         y1=p_threshold3,
                         line=dict(color="gray", width=2, dash="dash"))
                ]
            )
        else:
            # sort by 'neg_log10'
            fp_df.sort_values(by='neg_log10', ascending=False, inplace=True)
            fig.add_trace(
                go.Bar(x=fp_df.index, y=fp_df['neg_log10'].round(3), customdata=(fp_df['PR(>F)'] * 100).round(1),
                       texttemplate='%{y}<br>(p=%{customdata}%)'))
            fig.update_layout(title=title, showlegend=False, yaxis_title='-Log10(p)', shapes=[
                dict(
                    type="line",
                    x0=0, x1=1,
                    y0=p_threshold2, y1=p_threshold2, xref="paper", yref="y",
                    line=dict(color="orange", width=2, dash="dash")
                )
            ])
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
    num_fields = [field['name'] for field in fields if field['attr'] in ('conti', 'disc') and 'target' not in field]
    cat_fields = [field['name'] for field in fields if field['attr'] == 'cat']
    fig = go.Figure()

    method = 'quantile'
    if cfg.get('method'):
        method = cfg['method']

    metric = 'euclidean'
    if cfg.get('metric'):
        metric = cfg['metric']

    disp = 'pca'
    if cfg.get('disp'):
        disp = cfg['disp']

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
                                subplot_titles=[f'{n} ({outers[n].count()}/{len(df)})' for n in outers.columns],
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
            fig.update_layout(title=f'Method: {method}, abnormal features: {len(outers.columns)}', height=num_row * 400, showlegend=False)
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
                                subplot_titles=[f'{n} ({outers[n].count()}/{len(df)})' for n in outers.columns if n != 'type'],
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
            radius = threshold if threshold else 0.5
            min_s = cfg['min_samples'] if cfg.get('min_samples') else 5
            clf = DBSCAN(eps=radius, min_samples=min_s, metric=metric)
            clf.fit(df[num_fields])
            y_pred = [1 if i < 0 else 0 for i in clf.labels_]
            df['outlier'] = y_pred
        case 'svm':
            kernel = 'rbf'
            if cfg.get('kernel'):
                kernel = cfg['kernel']
            # One-Class SVM (classfication-based)
            # kernel: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
            cont_ratio = threshold if threshold else 0.03
            clf = OCSVM(nu=0.5, contamination=cont_ratio, kernel=kernel)
            clf.fit(df[num_fields])
            y_pred = clf.labels_
            df['outlier'] = y_pred

            # 提取异常点（ocsvm_anomaly == 1）
            anomaly = df[df['outlier'] != 0]
            background_data = shap.sample(df[num_fields], 100)
            # 使用 KernelExplainer（适用于 OCSVM）
            explainer = shap.KernelExplainer(clf.decision_function, background_data)
            shap_values = explainer.shap_values(df[num_fields], nsamples=100)
            shap_ocsvm_df = pd.DataFrame(shap_values, columns=num_fields, index=df.index)
            shap_anomalies = shap_ocsvm_df.loc[anomaly.index]
            df['suspect'] = shap_anomalies.abs().idxmax(axis=1)
            df['suspect'] = df['suspect'].fillna('')
        case 'knn':
            #  K-Nearest Neighbors (distance-based)
            # ['braycurtis', 'canberra', 'chebyshev',
            # 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
            # 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
            # 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
            # 'sqeuclidean', 'yule']
            # 'correlation' does not work
            # contamination: proportion of outliers
            cont_ratio = threshold if threshold else 0.03
            clf = KNN(contamination=cont_ratio, n_neighbors=5, method='mean', metric=metric)
            clf.fit(df[num_fields])
            # 1: outlier
            y_pred = clf.labels_
            df['outlier'] = y_pred
        case 'lof':
            # Local Outlier Factor(局部利群因子, density-based)
            # ['braycurtis', 'canberra', 'chebyshev',
            # 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
            # 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
            # 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
            # 'sqeuclidean', 'yule']
            cont_ratio = threshold if threshold else 0.03
            clf = LOF(contamination=cont_ratio, n_neighbors=10, metric=metric)
            y_pred = clf.fit_predict(df[num_fields])
            df['outlier'] = y_pred
        case 'cof':
            # Connectivity-Based Outlier Factor (COF, LOF的变种, density-based)
            cont_ratio = threshold if threshold else 0.03
            clf = COF(contamination=cont_ratio, n_neighbors=15)
            clf.fit(df[num_fields])
            y_pred = clf.predict(df[num_fields])
            df['outlier'] = y_pred
        case 'iforest':
            # Isolation Forest(孤立森林, tree-based)
            # unsupervised, global outlier detection
            # contamination: percentage of outliers
            # n_estimators: total of trees
            cont_ratio = threshold if threshold else 0.03
            clf = IForest(contamination=cont_ratio, n_estimators=100, max_samples='auto')
            clf.fit(df[num_fields])
            y_pred = clf.labels_
            df['outlier'] = y_pred
            # use shap to explain the outliers
            anomaly = df[df['outlier'] != 0]
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(df[num_fields])
            shap_df = pd.DataFrame(
                shap_values[anomaly.index],
                index=anomaly.index,
                columns=num_fields
            )
            shap_anomalies = shap_df.loc[anomaly.index]
            df['suspect'] = shap_anomalies.abs().idxmax(axis=1)
            df['suspect'] = df['suspect'].fillna('')
        case 'som':
            # self-organizing map(自组织映射算法)
            # 是一种无监督学习算法，用于对高维数据进行降维和聚类分析
            # 无监督，非线性
            cont_ratio = threshold if threshold else 0.03
            som = MiniSom(15, 15, len(num_fields), sigma=3, learning_rate=0.5, activation_distance=metric, neighborhood_function='triangle')
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
            df['outlier'] = y_pred
        case 'vae':
            # AutoEncoder(自编码器, unsupervised, neural network)
            epoch = cfg.get('epoch', 1)
            batch = cfg.get('batch', 32)
            cont_ratio = threshold if threshold else 0.03
            if torch.cuda.is_available():
                device = torch.device("cuda")
                auto_encoder = vae.VAE(epoch_num=epoch, batch_size=batch, contamination=cont_ratio, device=device)
            else:
                auto_encoder = vae.VAE(epoch_num=epoch, batch_size=batch, contamination=cont_ratio)
            auto_encoder.fit(df[num_fields])
            # 1: outlier
            y_pred = auto_encoder.predict(df[num_fields])
            df['outlier'] = y_pred
        case '_':
            return fig

    # display outliers by t-SNE/UMAP chart
    dim = cfg.get('dim', 2)
    if len(num_fields)==1:
        # time series with one value field
        date_fields = [field['name'] for field in fields if field['attr'] == 'date' or 'timeline' in field]
        df.set_index(date_fields[0], inplace=True)
        if len(date_fields) > 0:
            y_df = pd.DataFrame(df[num_fields[0]]*y_pred)
            y_df = y_df[y_df[num_fields[0]]>0]
            fig.add_trace(go.Scatter(x=df.index, y=df[num_fields[0]], name=num_fields[0], hovertemplate='%{y}<extra></extra>'))
            fig.add_trace(go.Scatter(x=y_df.index, y=y_df[num_fields[0]], name='outlier', marker=dict(color="red"),
                                     mode='markers', hovertemplate='%{y}<extra></extra>'))
            fig.update_layout(title=f'Outliers: {sum(y_pred)}/{len(y_df)}', hovermode='x')
        return fig
    else:
        dim = min(dim, len(num_fields))

    labels = None
    # visualization solution
    if disp == 'umap':
        # UMAP: Uniform Manifold Approximation and Projection
        data = umap.UMAP(n_components=dim, metric=metric).fit_transform(df[num_fields])
    elif disp == 'tsne':
        # t-distributed Stochastic Neighbor Embedding
        data = manifold.TSNE(n_components=dim, metric=metric).fit_transform(df[num_fields])
    else:
        # PCA (Principal Component Analysis)
        pca = PCA(n_components=dim)
        data = pca.fit_transform(df[num_fields])
        labels = {
            f'd{i}': f"PCA{i} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }

    # add pca data to original dataset
    pca_df = pd.DataFrame(data, columns=[f'd{i}' for i in range(dim)])
    df = pd.concat([df, pca_df], axis=1)
    vis_cols = pca_df.columns.tolist()
    # point type: inner or outlier for displaying
    df['point_type'] = df['outlier'].map({0: 'inner', 1: 'outlier'})
    # reset index for label displaying
    df.reset_index(inplace=True)
    title = f'Method: {method}, Outliers: {sum(y_pred)}/{len(y_pred)}'

    # tooltip title
    hover_name = cfg.get('label', 'index')
    hover_data = valid_fields
    if 'suspect' in df.columns:
        # show suspect field on tooltip
        hover_data.append('suspect')
    df['label_text'] = ''
    if cfg.get('label') is not None:
        # show outlier's label on scatter plot
        df['label_text'] = df[hover_name].astype(str).where(df['outlier'] != 0, '')
        if 'suspect' in df.columns:
            # combin column name
            df['label_text'] = df.apply(lambda row: row['label_text'] + f"({row['suspect']})" if row['suspect'] != '' else row['label_text'], axis=1)

    if dim > 3:
        mean_v = np.mean(data, axis=0)
        std_v = np.std(data, axis=0)
        th_upper = mean_v + 8 * std_v
        th_lower = mean_v + 2 * std_v
        mean_v = mean_v + 5 * std_v

        max_mean = max(mean_v)
        off_set = []
        for i in range(len(mean_v)):
            if mean_v[i] != max_mean:
                off_set.append(math.floor(max_mean - mean_v[i]))
            else:
                off_set.append(0)

        mean_v = mean_v + np.array(off_set)
        th_upper = th_upper + np.array(off_set)
        th_lower = th_lower + np.array(off_set)

        # outliers
        out_df = df[df['outlier'] != 0]
        org_df = out_df[num_fields]
        out_df[vis_cols] = out_df[vis_cols] + pd.Series(mean_v, index=vis_cols)
        df_melted = pd.melt(out_df,
                            id_vars=['index'], # keep index as id
                            value_vars=vis_cols, # put value of [d0, d1, d2, d3] into one column(value_name)
                            var_name='feature', # value: d0, d1, d2, d3
                            value_name='value') # contain value of [d0, d1, d2, d3]
        df_melted.set_index('index', inplace=True, drop=False)
        # merge original num_fields into df_melted for displaying
        df_melted = df_melted.join(org_df, how='left')
        fig = px.line_polar(df_melted, r='value', theta='feature', color='index', line_close=True,
                            hover_data=num_fields, hover_name=hover_name)

        # mean
        fig.add_trace(go.Scatterpolar(
            r=mean_v.tolist() + [mean_v[0]],
            theta=vis_cols + [vis_cols[0]],
            line=dict(color='blue', width=2),
            name='Mean'
        ))

        # upper threshold
        fig.add_trace(go.Scatterpolar(
            r=th_upper.tolist() + [th_upper[0]],
            theta=vis_cols + [vis_cols[0]],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='99%'
        ))

        # lower threshold
        fig.add_trace(go.Scatterpolar(
            r=th_lower.tolist() + [th_lower[0]],
            theta=vis_cols + [vis_cols[0]],
            fill='toself',
            fillcolor='rgba(255,255,255,1)',
            line=dict(color='rgba(0,0,0,0)'),
            name=''
        ))
    elif dim == 3:
        fig = px.scatter_3d(df, x='d0', y='d1', z='d2', size=[5]*len(df), color=df['point_type'], size_max=10, opacity=1,
                            color_discrete_map={'inner': "rgba(0, 204, 150, 0.7)", 'outlier': 'rgba(255, 0, 0, 1)'},
                            category_orders={'outlier': [0, 1]}, labels=labels,
                            hover_data=hover_data, hover_name=hover_name)
    else:
        fig = px.scatter(df, x='d0', y='d1', color='point_type', labels=labels, text='label_text',
                         color_discrete_map={'inner': "rgba(0, 204, 150, 0.5)", 'outlier': 'rgba(255, 0, 0, 1)'},
                         category_orders={'outlier': [0, 1]}, hover_name=hover_name,
                         hover_data=valid_fields)

    fig.update_traces(textposition="bottom center")
    fig.update_layout(title=title, legend_title_text='', legend=dict(xanchor="right", yanchor="top", x=0.99, y=0.99))
    return fig


"""
Variance rank
"""
def plt_stat_var(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] in ('conti', 'disc')]
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
calculate data volume by categorical fields
"""
def plt_dist_volume(cfg, df, fields):
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
                fig.add_trace(go.Funnel(x=cat_df['count'], y=cat_df[name], name=name, texttemplate = "%{percentTotal:.1%}",),
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
        fig.layout.annotations[idx].text = f'{name} ({len(cat_df)})'
    if cfg.get('funnel'):
        # avoid float on y axes
        fig.update_yaxes(type='category')

    fig.update_layout(title='Data volume by category', height=row_h*num_row, showlegend=False, hovermode='x', barmode='stack')
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

    xerror = cfg.get('xerror')
    yerror = cfg.get('yerror')
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
        # sort by category
        df.sort_values(by=cat, inplace=True, key=lambda x: x.str.lower())
        # category by colors
        fig = px.scatter(df, x=xf, y=yf, color=cat, trendline=trend, trendline_scope="overall",
                         trendline_options=dict(frac=frac), facet_row=facet, marginal_x=marg, marginal_y=marg,
                         error_x=xerror, error_y=yerror)
    else:
        fig = px.scatter(df, x=xf, y=yf, trendline=trend, trendline_options=dict(frac=frac),
                         facet_row=facet, marginal_x=marg, marginal_y=marg, error_x=xerror, error_y=yerror)

    fig.update_layout(title=f'Scatter plot ({xf} X {yf})', legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    return fig


"""
Scatter Matrix without upperhalf and diagonal
"""
def plt_corr_scatters(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] == 'conti'] + \
                  [field['name'] for field in fields if field['attr'] == 'disc']

    title = 'Scatter matrix'
    cat = None
    if cfg.get('cf'):
        cat = cfg['cf']
        title += f', Category field: {cat}'
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
    fig.update_layout(title=title, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
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
        fig.update_layout(title='Pair plot', width=1600, height=800)
        return fig
    else:
        return None



"""
CCM: Correlation Coefficient Matrix
"""
def plt_corr_ccm(cfg, df, fields):
    # 相关系数也可以看成协方差：一种剔除了两个变量量纲影响、标准化后的特殊协方差。
    # Pearson 系数用来检测两个连续型变量之间线性相关程度，要求这两个变量分别分布服从正态分布；仅检测线性关系,容易受异常值影响
    # Spearman系数不假设连续变量服从何种分布，如果是顺序变量(Ordinal)，推荐使用Spearman。不要求正态分布,对异常值不敏感,捕捉单调非线性关系
    # Kendall 用于检验连续变量和类别变量间的相关性
    # 卡方检验(Chi-squared Test)，检验类别变量间的相关性
    num_fields = [field['name'] for field in fields if field['attr'] in ['conti', 'disc']]
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
    fig.update_layout(title=f'Correlation coefficient matrix, coeff: {coeff}', xaxis_showgrid=False,
                      yaxis_showgrid=False, yaxis_autorange='reversed', template='plotly_white')
    return fig

"""
COV: Covariance Matrix
"""
def plt_corr_cov(cfg, df, fields):
    # Covariance Matrix: 协方差矩阵
    num_fields = [field['name'] for field in fields if field['attr'] in ['conti', 'disc']]
    cat_fields = [field['name'] for field in fields if field['attr'] == 'cat']
    if cfg.get('num'):
        df_corr = df[[n for n in num_fields]].cov(numeric_only=True).round(2)
    else:
        df_corr = df[[n for n in num_fields + cat_fields]].corr(numeric_only=False).round(2)

    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    viz_corr = df_corr.mask(mask).dropna(how='all').dropna(axis=1, how='all')
    viz_corr = viz_corr.replace({np.nan: ''})

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=viz_corr, x=viz_corr.columns, y=viz_corr.index,
                             hoverinfo="none", colorscale=px.colors.diverging.RdBu, text=viz_corr.values,
                             texttemplate="%{text}", zmin=-1, zmax=1, ygap=1, xgap=1))
    fig.update_layout(title='Covariance matrix', xaxis_showgrid=False, yaxis_showgrid=False, yaxis_autorange='reversed', template='plotly_white')
    return fig


"""
Parallel curves
"""
def plt_corr_parallel(cfg, df, fields):
    num_fields = [field['name'] for field in fields if field['attr'] in ('conti', 'disc')]
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
    num_fields = [field['name'] for field in fields if field['attr'] in ('conti', 'disc')]

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

    title = f'Correlation coefficient matrix, type: {method}'
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
    else:
        fig.update_layout(title=title)
    return fig


"""
clustering detection
K-Means
"""
def plt_cluster_kmeans(cfg, df, fields):
    valid_fields = [field for field in fields if 'omit' not in field]
    target_field = [field['name'] for field in valid_fields if 'target' in field]
    num_fields = [field['name'] for field in valid_fields if field['attr'] in ('conti', 'disc') and 'target' not in field]
    cat_fields = [field['name'] for field in valid_fields if field['attr'] == 'cat']
    fig = go.Figure()

    method = cfg.get('method', 'kmeans')
    n_clusters = cfg.get('clusters', 2)
    max_iter = cfg.get('iterations', 300)
    tol = cfg.get('tol', 0.0001)
    disp = cfg.get('disp', 'pca')
    metric = cfg.get('metric', 'euclidean')

    cluster_ids = None
    match method:
        case 'dbscan':
            # Density-Based Spatial Clustering of Applications with Noise，具有噪声的基于密度的聚类方法(cluster-based)
            #  ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’,
            #  ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
            #  ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’
            # 'minkowski' does not work
            radius = 0.5
            min_s = cfg['min_samples'] if cfg.get('min_samples') else 5
            clst = DBSCAN(eps=radius, min_samples=min_s, metric=metric)
            clst.fit(df[num_fields])
            cluster_ids = clst.labels_
        case 'kmeans':
            clst = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iter, tol=tol)
            clst.fit(df[num_fields])
            cluster_ids = clst.labels_
        case '_':
            return fig

    # display clusters by t-SNE
    dim = cfg.get('dim', 2)
    labels = None
    # visualization solution
    if disp == 'tsne':
        # t-distributed Stochastic Neighbor Embedding
        data = manifold.TSNE(n_components=dim, metric=metric, perplexity=min(30, len(df)-1)).fit_transform(df[num_fields])
    else:
        # PCA (Principal Component Analysis)
        pca = PCA(n_components=dim)
        data = pca.fit_transform(df[num_fields])
        labels = {
            f'd{i}': f"PCA{i} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }

    # build dataframe based on data of 2 dim (d0, d1)
    vis_df = pd.DataFrame(data, columns=[f'd{i}' for i in range(dim)])
    # merge vis data into original data
    df = pd.concat([df, vis_df], axis=1)

    # add cluster id
    df['cluster'] = cluster_ids
    # convert index to a column
    df.reset_index(inplace=True)
    label_cluster = cfg.get('cluster')
    hover_name = cfg.get('label', 'index')
    df['label_text'] = ''
    if cfg.get('label') is not None:
        if label_cluster is None:
            # show labels on scatter
            df['label_text'] = df[hover_name]
        else:
            # show labels on scatter plot for a specific cluster
            df['label_text'] = df[hover_name].astype(str).where(df['cluster'] == label_cluster, '')
    # convert cluster id to string for legend displaying
    df['cluster'] = df['cluster'].astype(str)
    # sort by cluster id and index for displaying (legend order)
    df.sort_values(by=['cluster', 'index'], ascending=[True, True], inplace=True)
    u_cids = df['cluster'].value_counts()
    u_str = '(' + ', '.join([f"{k}:{v}" for k, v in u_cids.items()]) + ')'

    # build plotly figure
    title = f'Clustering method: {method} {u_str}'
    fig = px.scatter(df, x='d0', y='d1', color='cluster', labels=labels, hover_name=hover_name,
                     hover_data=num_fields+target_field, text='label_text')

    fig.update_traces(textposition="bottom center")
    fig.update_layout(title=title, legend=dict(xanchor="right", yanchor="top", x=0.99, y=0.99))
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
plt time series lines
{"pid": "ts", "tf": "time", "period": "YE", "vf": "kpi", "agg": "mean", "cat": "country", "solo": true, "connected": true}
"""
def plt_ts_series(df, cfg, fields):
    fig = go.Figure()
    ts_name = df.index.name

    # category/value field
    cat_field = cfg.get('cat')
    v_fields = cfg['vf']

    line_with_gaps = cfg.get('gap', False)
    connected = True
    if cfg.get('connected'):
        connected = cfg['connected']

    if line_with_gaps:
        # get all cat names with missing values
        cat_values = df[df[v_fields].isna().any(axis=1)][cat_field].unique().astype(str)

    title = f'Time series - {cfg["agg"]}, [{df.index.min()} ~ {df.index.max()}]'
    if cat_field:
        df.reset_index(inplace=True)
        cat_total = df[cat_field].nunique()
        if len(v_fields) == 1:
            if cfg.get('solo'):
                fig = px.line(df, x=ts_name, y=v_fields[0], facet_col=cat_field, facet_col_wrap=2,
                              facet_row_spacing=min(0.05,1/cat_total), height=800 if cat_total < 3 else 300)
                fig.update_layout(legend_title_text=f'{cat_field} ({cat_total})',
                                  height=800 if cat_total < 3 else (math.ceil(cat_total/2) * 300))
            else:
                fig = px.line(df, x=ts_name, y=v_fields[0], color=cat_field)
                fig.update_layout(legend_title_text=f'{cat_field} ({cat_total})')
            fig.for_each_xaxis(lambda x: x.update(title=''))
        else:
            # put curves on separated charts by value fields
            cols = 1 if len(v_fields) < 4 else 2
            rows = math.ceil(len(v_fields) / cols)
            fig = make_subplots(rows=rows, cols=cols, row_heights=[800 if rows == 1 else 300 for n in range(rows)],
                                horizontal_spacing=0.05, vertical_spacing=min(1/rows, 0.1))
            for idx, vfield in enumerate(v_fields):
                # build lines by category
                px_line = px.line(df, x=ts_name, y=vfield, color=cat_field)
                # sort lines by category name
                sorted_lines = sorted(px_line.data, key=lambda x: x['name'])
                # all sub plots have same legend. show first legend only
                [fig.add_trace(go.Scatter(x=line['x'], y=line['y'], name=line['name'], connectgaps=connected, showlegend=True if idx==0 else False),
                               (idx // cols) + 1, (idx % cols) + 1) for line in sorted_lines]
            [fig.update_yaxes(row=(i // cols) + 1, col=(i % cols) + 1, title_text=nf) for i, nf in enumerate(v_fields)]
            fig.update_xaxes(matches='x')
            fig.update_layout(legend_title_text=f'{cat_field} ({cat_total})', height=800 if rows == 1 else (rows * 300))
    else:
        if cfg.get('solo'):
            # put curves on separated charts by value fields
            cols = 1 if len(v_fields) < 4 else 2
            rows = math.ceil(len(v_fields) / cols)
            fig = make_subplots(rows=rows, cols=cols, row_heights=[800 if rows == 1 else 300 for n in range(rows)],
                                horizontal_spacing=0.05, vertical_spacing=min(1/rows, 0.1))
            [fig.add_trace(go.Scatter(x=df.index, y=df[nf], name='', connectgaps=connected),
                           (i // cols) + 1, (i % cols) + 1) for i, nf in enumerate(v_fields)]
            [fig.update_yaxes(row=(i // cols) + 1, col=(i % cols) + 1, title_text=nf) for i, nf in enumerate(v_fields)]
            fig.update_xaxes(matches='x')
            fig.update_layout(showlegend=False, height=800 if rows == 1 else (rows * 300))
        else:
            # put all curves on one chart
            [fig.add_trace(go.Scatter(x=df.index, y=df[nf], name=nf, connectgaps=connected)) for nf in v_fields]
            if (len(v_fields) == 1):
                # specific y axis label
                fig.update_yaxes(title_text=v_fields[0])
            else:
                # general y axis label
                fig.update_yaxes(title_text='value')
    fig.update_layout(title=title, hovermode='x')
    return fig


"""
plt ts trending line chart
# ols(线性普通最小二乘), lowess(局部加权线性回归), rolling(移动平均线), ewm(指数加权移动平均), expanding(扩展窗)
{"pid": "ts", "ts": "date", "vf": "open",  "period": "M", "agg": "mean", "frac": 0.6}
"""
def plt_ts_trend(df, cfg, fields):
    # value field and period are mandatory
    if cfg.get('vf') is None or cfg.get('period') is None:
        return None

    method = cfg.get('method', 'ols')
    vfield = cfg['vf']
    cfield = cfg.get('cat')
    connected = False
    if cfg.get('connected'):
        connected = cfg['connected']

    # aggregated by period (YS,QS,MS,W,D,h,min,s)
    if cfg['period'].startswith('Y'):
        # show year only (don't show month and day for Y)
        df.index = df.index.strftime('%Y')
    elif cfg['period'].startswith('QQ'):
        # convert to period for getting quarters
        df.index = df.index.to_period('Q')
        df.index = df.index.strftime('%Y-%q')

    max_win = len(df)
    frac = 0.667  # [0, 1]
    if cfg.get('frac'):
        frac = cfg['frac']
        if frac < 0:
            frac = 0
        elif frac > 1:
            frac = 1

    win = math.floor(frac * max_win)
    if win <= 0:
        win = 1

    cat_pv = dict()
    match method:
        case 'ols':
            # 线性
            if cfg.get('diff'):
                fig = px.scatter(df, x=df.index, y=vfield, color=cfield, trendline='ols')
            else:
                fig = px.scatter(df, x=df.index, y=vfield, facet_col=cfield, facet_col_wrap=2, trendline='ols',
                             trendline_color_override='orange')
                if cfield:
                    df['ts_min'] = (df.index - df.index.min()).total_seconds() / 60
                    linear_model = smf.ols(f'{vfield} ~ ts_min * {cfield}', data=df).fit()
                    for k, v in linear_model.pvalues.items():
                        if k.startswith('ts_min:'):
                            sub_k = k.split('T.')[1]
                            cat_k = sub_k[0:-1]
                            if v < 0.001:
                                cat_pv[cat_k] = '***'
                            elif v < 0.01:
                                cat_pv[cat_k] = '**'
                            elif v < 0.05:
                                cat_pv[cat_k] = '*'
                            elif v < 0.1:
                                cat_pv[cat_k] = '.'
                            else:
                                cat_pv[cat_k] = '-'
        case  'lowess':
            # The fraction of the data used when estimating each y-value.
            # 平滑
            if cfg.get('diff'):
                fig = px.scatter(df, x=df.index, y=vfield, color=cfield, trendline='lowess',
                                 trendline_options=dict(frac=frac))
            else:
                fig = px.scatter(df, x=df.index, y=vfield, facet_col=cfield, facet_col_wrap=2, trendline='lowess',
                             trendline_options=dict(frac=frac), trendline_color_override='orange')
        case  'rolling':
            # 中心滞后，权重相同
            if cfg.get('diff'):
                fig = px.scatter(df, x=df.index, y=vfield, color=cfield, trendline='rolling',
                                 trendline_options=dict(window=win, min_periods=1))
            else:
                fig = px.scatter(df, x=df.index, y=vfield, facet_col=cfield, facet_col_wrap=2, trendline='rolling',
                             trendline_options=dict(window=win, min_periods=1), trendline_color_override='orange')
        case 'ewm':
            # 中心滞后，权重衰减
            if cfg.get('diff'):
                fig = px.scatter(df, x=df.index, y=vfield, color=cfield, trendline='ewm',
                                 trendline_options=dict(halflife=win))
            else:
                fig = px.scatter(df, x=df.index, y=vfield, facet_col=cfield, facet_col_wrap=2, trendline='ewm',
                             trendline_options=dict(halflife=win), trendline_color_override='orange')
        case 'polynomial':
            # show original data
            fig = px.scatter(df, x=df.index, y=vfield, facet_col=cfield, facet_col_wrap=2)
            # it is OLS when degree=1
            forecaster = PolynomialTrendForecaster(degree=int(1 // frac))
            if cfield:
                df['trend'] = None
                u_cats = df[cfield].unique().astype(str)
                for cat in u_cats:
                    cat_df = df[df[cfield] == cat][vfield]
                    prange = pd.date_range(cat_df.index.min(), periods=len(cat_df), freq=cfg['period'])
                    trend_v = forecaster.fit(cat_df).predict(fh=prange)
                    df.loc[df[cfield] == cat, 'trend'] = trend_v.tolist()
                lines = px.line(df, x=df.index, y='trend', facet_col=cfield, facet_col_wrap=2, hover_data=[],
                                    color_discrete_sequence=['orange'])
                fig.add_traces(lines.data)
            else:
                prange = pd.date_range(df.index.min(), periods=len(df), freq=cfg['period'])
                df['poly'] = forecaster.fit(df[vfield]).predict(fh=prange)
                fig.add_scatter(x=df.index, y=df['trend'], name='Polynomial', showlegend=False, line_color='orange')

    if cfg.get('diff') and cfield:
        # add all trendlines to one chart
        for trace in fig.data:
            # extract data of trendline from trace
            if 'trendline' in trace.hovertemplate:
                df.loc[df[cfield] == trace.name, 'trend'] = trace.y
        # show all trendlines without original data
        fig = px.line(df, x=df.index, y="trend", color=cfield)
    elif connected:
        fig.update_traces(mode='lines')

    if cfield and method == 'ols':
        for i, ann in enumerate(fig.layout.annotations):
            cat_n = ann.text.split('=')[-1]
            if cat_n in cat_pv:
                ann.text += f' ({cat_pv[cat_n]})'
            else:
                ann.text += ' (reference)'

    title = f'Ts trending of {method}, [{df.index.min()} ~ {df.index.max()}]'
    fig.for_each_xaxis(lambda x: x.update(title=''))
    fig.update_layout(title=title, yaxis_title=vfield, hovermode='x')
    return fig


"""
plt ts difference chart
# 差分，与前面lag个周期值的差，可见指标增长或下降
# order, 差分次数(阶数)
{"pid": "ts", "ts": "time", "period": "YE", "agg": "mean", "solo": false, "vf": ["dena72"], "lag": 1, "step": 1}
"""
def plt_ts_diff(df, cfg, fields):
    fig = go.Figure()
    # cfg['vf'] is a list
    if cfg.get('vf') is None or cfg.get('period') is None:
        return fig

    lag = cfg.get('lag', 1)
    order = cfg.get('order', 1)

    vf = cfg.get('vf')
    if cfg.get('period') and cfg.get('period').startswith('Y'):
        # show year only (don't show month and day for Y)
        df.index = df.index.strftime('%Y')

    for i in range(order):
        df[vf] = df[vf].diff(periods=lag)
    if cfg.get('solo'):
        # put curves on separated charts
        rows = len(vf)
        fig = make_subplots(rows=rows, cols=1, subplot_titles=vf, row_heights=[300 for n in range(rows)],
                            horizontal_spacing=0.05, vertical_spacing=0.2 / rows)
        [fig.add_trace(go.Bar(x=df.index, y=df[nf], name=''), i + 1, 1) for i, nf in enumerate(vf)]
        fig.update_xaxes(matches='x')
        fig.update_layout(showlegend=False, height=rows * 300)
    else:
        # put all curves on one chart
        [fig.add_trace(go.Bar(x=df.index, y=df[nf], name=nf, hovertemplate='%{y}<extra></extra>'))
         for nf in vf]

    # 间隔为lag的order阶差分
    title = f'{order} order difference lag of {lag}, [{df.index.min()} ~ {df.index.max()}]'
    fig.update_layout(title=title, hovermode='x')
    return fig


"""
plt ts frequency chart
尽量选择一个完整的周期，如2020-08-05 到2-24-08-05
{"pid": "ts", "ts": "time", "field": "dena74", "agg": "sum"}
"""
def plt_ts_freq(df, cfg, fields):
    fig = go.Figure()
    if cfg.get('vf') is None or (isinstance(cfg.get('vf'), list) and len(cfg['vf']) != 1):
        return fig

    vfield = cfg['vf']  # value field
    if isinstance(cfg.get('vf'), list):
        vfield = cfg['vf'][0]

    # agg: sum, mean, median, min, max, count
    agg = 'mean'
    if cfg.get('agg'):
        agg = cfg['agg']

    ts_name = df.index.name
    title = f"{vfield} frequency, [{df.index.min()} ~ {df.index.max()}]"
    fig = make_subplots(rows=3, cols=2, specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
                        subplot_titles=['Quarterly', 'Monthly', 'Weekly', 'Hourly', 'Daily'])

    # Quarterly
    ts_df = pd.DataFrame({ts_name: df.index.quarter, vfield: df[vfield]})
    ts_df.set_index(ts_name, inplace=True)
    gp_df = ts_df.groupby(ts_name).agg(agg).round(3)
    per_df = pd.DataFrame(columns=[ts_name], data=range(1, 5, 1))
    merged_df = pd.merge(per_df, gp_df, left_on=ts_name, right_index=True, how="left")
    fig.add_trace(go.Bar(x='Q' + merged_df[ts_name].astype('string'), y=merged_df[vfield], name='',
                         text=merged_df[vfield]), 1, 1)

    # Monthly
    ts_df = pd.DataFrame({ts_name: df.index.strftime('%b'), vfield: df[vfield]})
    ts_df.set_index(ts_name, inplace=True)
    gp_df = ts_df.groupby(ts_name).agg(agg).round(3)
    per_df = pd.DataFrame(columns=[ts_name], data=pd.date_range(start='2024-01-01', end='2024-12-01', freq='MS')
                          .strftime('%b').to_list())
    merged_df = pd.merge(per_df, gp_df, left_on=ts_name, right_index=True, how="left")
    fig.add_trace(go.Bar(x=merged_df[ts_name], y=merged_df[vfield], name='', text=merged_df[vfield]), 1, 2)

    # Weekly
    ts_df = pd.DataFrame({ts_name: df.index.strftime('%a'), vfield: df[vfield]})
    ts_df.set_index(ts_name, inplace=True)
    gp_df = ts_df.groupby(ts_name).agg(agg).round(3)
    per_df = pd.DataFrame(columns=[ts_name],
                          data=pd.date_range(start='2024-09-01', end='2024-09-07', freq='D')
                          .strftime('%a').to_list())
    merged_df = pd.merge(per_df, gp_df, left_on=ts_name, right_index=True, how="left")
    fig.add_trace(go.Bar(x=merged_df[ts_name], y=merged_df[vfield], name='', text=merged_df[vfield]), 2, 1)

    # Hourly
    ts_df = pd.DataFrame({ts_name: df.index.hour, vfield: df[vfield]})
    ts_df.set_index(ts_name, inplace=True)
    gp_df = ts_df.groupby(ts_name).agg(agg).round(3)
    per_df = pd.DataFrame(columns=[ts_name], data=range(0, 24, 1))
    merged_df = pd.merge(per_df, gp_df, left_on=ts_name, right_index=True, how="left")
    fig.add_trace(go.Bar(x=merged_df[ts_name], y=merged_df[vfield], name='', text=merged_df[vfield]), 2, 2)

    # Daily
    ts_df = pd.DataFrame({ts_name: df.index.day, vfield: df[vfield]})
    ts_df.set_index(ts_name, inplace=True)
    gp_df = ts_df.groupby(ts_name).agg(agg).round(3)
    per_df = pd.DataFrame(columns=[ts_name], data=range(1, 32, 1))
    merged_df = pd.merge(per_df, gp_df, left_on=ts_name, right_index=True, how="left")
    fig.add_trace(go.Bar(x=merged_df[ts_name], y=merged_df[vfield], name='', text=merged_df[vfield]), 3, 1)

    fig.update_xaxes(type='category')
    fig.update_layout(title=title, yaxis_title=f'{vfield} ({agg})', showlegend=False)
    return fig


"""
plt ts compare chart
# Y: year, q: quarter, m: month, w: week, d: day, H: hour, M: min, S: sec. 
# refer to https://pandas.pydata.org/docs/reference/api/pandas.Period.strftime.html
{"pid": "ts", "ts": "time", "field": "dena74", "group": "m", "period": "Y", "agg": "sum"}
"""
def plt_ts_compare(df, cfg, fields):
    fig = go.Figure()
    if cfg.get('vf') is None or cfg.get('groupby') is None or cfg.get('period') is None:
        return fig

    # agg: sum, mean, median, min, max, count
    agg = cfg.get('agg', 'mean')
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

    # why didn't I use hisFunc for agg - it has been done above
    fig = px.histogram(ts_df, x='ts_' + group, y=vfield, color='ts_' + period, barmode='group')
    fig.update_xaxes(type='category')
    title = f'Contemporary comparison, [{df.index.min()} ~ {df.index.max()}]'
    fig.update_layout(title=title, xaxis_title='', yaxis_title=f'{vfield} ({agg})', legend_title='')
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
def plt_ts_acf(df, cfg, fields):
    fig = go.Figure()
    if cfg.get('period') is None or cfg.get('vf') is None:
        return fig

    vfield = cfg['vf']
    title = f'{vfield} stationarity test, [{df.index.min()} ~ {df.index.max()}]'
    if cfg.get('order') is not None and cfg['order'] > 0:
        order = cfg['order']
        title = f'{vfield} stationarity after {order} order diff, [{df.index.min()} ~ {df.index.max()}]'
        for i in range(order):
            df[vfield] = df[vfield].diff(periods=1)
        # drop null rows
        df = df.dropna()

    # Can only compute partial correlations for lags up to 50% of the sample size
    lag = cfg.get('lag', 10)
    pacf_limit = math.floor(len(df) / 2)
    if lag > pacf_limit:
        lag = pacf_limit - 1

    # adf test and kpss test to detect if it is stationary
    adf_test = adfuller(df[vfield], autolag='AIC')
    p_val1 = adf_test[1]
    kpss_test = kpss(df[vfield], regression='c')
    p_val2 = kpss_test[1]

    test_result = 'non-stationary'
    if (p_val1 < 0.05) and (p_val2 > 0.05):
        test_result = 'stationary'

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1,
                        subplot_titles=[f'AutoCorrelation ({test_result})', f'Partial AutoCorrelation ({test_result})'])
    # Auto Correlation Function(自相关)
    acf_df, confint = acf(df[vfield], nlags=lag, alpha=0.05)
    acf_df = acf_df.round(3)
    # 95% confidence interval
    confint_lower = confint[:, 0] - acf_df
    confint_upper = confint[:, 1] - acf_df
    lags = np.arange(len(acf_df))
    fig.add_trace(go.Scatter(
        x=np.concatenate([lags, lags[::-1]]),
        y=np.concatenate([confint_upper, confint_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95%',
        showlegend=True
    ), 1, 1)
    # add matchstick chart
    [fig.add_trace(go.Scatter(x=(x, x), y=(0, acf_df[x]), mode='lines', name='', line_color='black'), 1, 1)
                    for x in range(len(acf_df))]
    fig.add_trace(go.Scatter(x=np.arange(len(acf_df)), y=acf_df, name='', mode='markers', marker_color='red'), 1, 1)

    # partial Auto Correlation Function(偏自相关)
    # method: ols, yw, ywm, ld, ldb
    pacf_df, confint = pacf(df[vfield], nlags=lag, alpha=0.05)
    pacf_df = pacf_df.round(3)
    # 95% confidence interval
    confint_lower = confint[:, 0] - pacf_df
    confint_upper = confint[:, 1] - pacf_df
    fig.add_trace(go.Scatter(
        x=np.concatenate([lags, lags[::-1]]),
        y=np.concatenate([confint_upper, confint_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(255,165,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95%',
        showlegend=True
    ), 2, 1)
    # add matchstick chart
    [fig.add_trace(go.Scatter(x=(x, x), y=(0, pacf_df[x]), mode='lines', name='', line_color='black'), 2, 1)
     for x in range(len(pacf_df))]
    fig.add_trace(go.Scatter(x=np.arange(len(pacf_df)), y=pacf_df, name='', mode='markers', marker_color='red'), 2, 1)

    fig.add_hline(y=0, line_dash='dash', line_color='gray')

    fig.update_yaxes(row=1, col=1, title_text='coefficient')
    fig.update_yaxes(row=2, col=1, title_text='coefficient')
    fig.update_xaxes(row=2, col=1, title_text='Lags')
    fig.update_layout(title=title, showlegend=False, hovermode='x')
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
def plt_ts_mavg(df, cfg, fields):
    fig = go.Figure()
    if cfg.get('vf') is None:
        return fig

    vfield = cfg['vf']
    win = cfg.get('win', 3)

    min_per = win
    if isinstance(win, int):
        min_per = math.ceil(win/2)
    fig.add_trace(go.Scatter(x=df.index, y=df[vfield], name=vfield, connectgaps=True))
    df[vfield] = df[vfield].rolling(window=win, min_periods=min_per).mean()
    fig.add_trace(go.Scatter(x=df.index, y=df[vfield], name='SMA', connectgaps=True))
    if isinstance(win, int):
        # WMA and EMA don't support date win like '2M'
        df[vfield] = df[vfield].rolling(window=win).apply(lambda x: x[::-1].cumsum().sum() * 2 / win / (win + 1))
        fig.add_trace(go.Scatter(x=df.index, y=df[vfield], name='WMA', connectgaps=True))

        df[vfield] = df[vfield].ewm(span=win).mean()
        fig.add_trace(go.Scatter(x=df.index, y=df[vfield], name='EMA', connectgaps=True))

    title = f'Moving average, [{df.index.min()} ~ {df.index.max()}]'
    fig.update_layout(title=title, yaxis_title=f'{vfield} ({cfg["agg"]})', hovermode='x')
    return fig


"""
plt ts mean+std, box or violin chart
{"pid": "ts", "ts": "time", "field": "dena74", "period": "Q", "violin": true}
"""
def plt_ts_distribution(df, cfg, fields):
    fig = go.Figure()
    if cfg.get('period') is None or cfg.get('vf') is None:
        return fig

    period = cfg['period']
    vfield = cfg['vf']  # metrix field
    cfield = cfg.get('cat')
    outlier = cfg.get('outlier', False)
    title = f'Quantile distribution, [{df.index.min()} ~ {df.index.max()}]'

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
    elif period.startswith('h'):
        ts_format = '%Y-%m-%d %H'
    elif period.startswith('min'):
        ts_format = '%Y-%m-%d %H:%M'
    else:
        ts_format = '%Y-%m-%d %H:%M:%S'

    ts_name = df.index.name
    df.index = df.index.to_period(period)
    df.reset_index(inplace=True)
    if cfg.get('disp') is None or cfg['disp'] == 'mean':
        if cfield:
            # mean and std of every ts group
            mean_std = df.groupby(by=[cfield, ts_name], observed=True)[vfield].agg(['mean', 'std']).reset_index()
            # sort categories for facet displaying
            u_cats = mean_std[cfield].unique().astype(str)
            u_cats.sort()

            # convert ts to str
            mean_std[ts_name] = mean_std[ts_name].dt.strftime(ts_format)
            mean_std['mean'] = mean_std['mean'].round(3)
            mean_std['std'] = mean_std['std'].round(3)

            # show mean +std following ascending order
            fig = px.scatter(mean_std, x=ts_name, y='mean', error_y='std', facet_col=cfield, facet_col_wrap=2,
                             category_orders={cfield: u_cats}, labels={ts_name: '', 'mean': vfield})
            lines = px.line(mean_std, x=ts_name, y='mean', facet_col=cfield, facet_col_wrap=2, hover_data=[],
                            category_orders={cfield: u_cats}, color_discrete_sequence=['orange'], labels={ts_name: '', 'mean': vfield})
            fig.add_traces(lines.data)
        else:
            # mean and std of every ts group
            mean_std = df.groupby(by=ts_name, observed=True)[vfield].agg(['mean', 'std']).reset_index()
            # convert ts to str
            mean_std[ts_name] = mean_std[ts_name].dt.strftime(ts_format)
            mean_std['mean'] = mean_std['mean'].round(3)
            mean_std['std'] = mean_std['std'].round(3)
            # show mean + std
            fig = px.scatter(mean_std, x=ts_name, y='mean', error_y='std')
            fig.add_traces(px.line(mean_std, x=ts_name, y='mean', hover_data=[], color_discrete_sequence=['orange']).data)
    else:
        # keep all original data points for box/violin without ts resample
        df.reset_index(inplace=True)
        df[ts_name] = df[ts_name].dt.strftime(ts_format)
        if cfield:
            if cfg['disp'] == 'violin':
                # show violin following ascending order
                fig = px.violin(df, x=ts_name, y=vfield, facet_col=cfield, facet_col_wrap=2, box=True, labels={ts_name: ''},
                                points=('outliers' if outlier else False))
            else:
                # show box following ascending order
                fig = px.box(df, x=ts_name, y=vfield, facet_col=cfield, facet_col_wrap=2, labels={ts_name: ''},
                             points=('outliers' if outlier else False))
        else:
            if cfg['disp'] == 'violin':
                # show violin following ascending order
                fig = px.violin(df, x=ts_name, y=vfield, box=True, points=('outliers' if outlier else False))
            else:
                # show box following ascending order
                fig = px.box(df, x=ts_name, y=vfield, points=('outliers' if outlier else False))

    fig.update_layout(title=title, showlegend=False, xaxis_title='', hovermode='x')
    return fig


"""
plt ts cycle chart
{"pid": "ts", "ts": "date", "field": "open",  "period": "M", "agg": "mean", "algo": "psd"}
"""
def plt_ts_cycle(df, cfg, fields):
    fig = go.Figure()
    if cfg.get('vf') is None:
        return fig

    vfield = cfg['vf']
    period = cfg.get('period')
    if period is not None and period.startswith('Y'):
        # show year only (don't show month and day for Y)
        df.index = df.index.strftime('%Y')
    elif period is not None and period.startswith('Q'):
        # convert to period for getting quarters
        df.index = df.index.to_period('Q')
        df.index = df.index.strftime('%Y-Q%q')

    algo = 'psd'
    if cfg.get('algo'):
        algo = cfg['algo']

    title = f'Periodicity test by {algo}, [{df.index.min()} ~ {df.index.max()}]'
    # df.reset_index(inplace=True)
    match algo:
        case 'psd':
            # Periodogram(周期图), PSD(Power spectral density, 功率谱密度)
            # 傅里叶变换和频谱分析
            freq, power = periodogram(df[vfield])
            freq = np.round(freq, 4)
            power = np.round(power, 4)
            psd_df = pd.DataFrame({'freq': freq, 'power': power})
            # get top 5 by power
            top_power = psd_df.nlargest(5, 'power')
            top_power = top_power[top_power['power'] > 0]
            # get cycles
            top_power['cycle'] = np.round(1 / top_power['freq'], 1)
            # build chart
            fig.add_trace(
                go.Scatter(x=psd_df['freq'], y=psd_df['power'], name='PSD', hovertemplate='%{y}<extra></extra>'))
            fig.add_trace(
                go.Scatter(x=top_power['freq'], y=top_power['power'], text=top_power['cycle'], name='Cycle',
                           mode='markers+text', textposition='top right', hovertemplate='%{text}<extra></extra>'))
            fig.update_xaxes(title=f'Frequency (1/{period})')
            fig.update_yaxes(title=f'{vfield} (Power/Hz)')

        case '_':
            return fig

    fig.update_layout(title=title, showlegend=False, hovermode='x')
    return fig


"""
plt ts decomposition chart
加法模型：y（t）=季节+趋势+周期+噪音
乘法模型：y（t）=季节*趋势*周期*噪音
# doesn't support 'min' and 's'
{"pid": "ts", "ts": "time", "field": "dena74",  "period": "D", "agg": "mean", "algo": "stl", "robust": true}
"""
def plt_ts_decomp(df, cfg, fields):
    fig = go.Figure()
    if cfg.get('vf') is None or cfg.get('period') is None:
        return fig

    vfield = cfg['vf']
    period = cfg.get('period')
    seasonal_period = cfg.get('sp')

    if period.startswith('Y'):
        # show year only (don't show month and day for Y)
        df.index = df.index.strftime('%Y')
    elif period.startswith('Q'):
        # convert to period for getting quarters
        df.index = df.index.to_period('Q')
        df.index = df.index.strftime('%Y-Q%q')
    elif 'min' in period:
        df.index = df.index.strftime('%Y-%m-%d %H:%M')
        # sp must be specified if minute period
        if cfg.get('sp') is None:
            cfg['sp'] = 60
    elif 'T' in period:
        df.index = df.index.strftime('%Y-%m-%d %H:%M')
        if cfg.get('sp') is None:
            # sp must be specified if minute period
            custom_min = period.split('T')[0]
            custom_min = int(custom_min)
            cfg['sp'] = math.floor(60/custom_min)

    algo = 'stl'
    if cfg.get('algo'):
        algo = cfg['algo']

    robust = False
    if cfg.get('robust'):
        robust = True

    # detect trend
    ts_seq = df[vfield].dropna()
    X = np.arange(len(ts_seq)).reshape(-1, 1)
    y = ts_seq.values.reshape(-1, 1)
    lreg = LinearRegression().fit(X, y)
    slope = lreg.coef_[0][0]
    threshold = np.std(y) * 0.03
    trending = 'no significant trend'
    if slope > threshold:
        trending = 'increasing trend'
    elif slope < -threshold:
        trending = 'decreasing trend'
    sub_titles = [f'Trend Component ({trending})']

    if cfg.get('sp'):
        # user defined seasonal_period
        seasonal_period = cfg['sp']
        sub_titles.append(f'Seasonal Component (custom period: {seasonal_period}{period})')
    else:
        # find best period
        resids = []
        for p in range(2, 31):
            try:
                res = STL(df[vfield], period=p, robust=True).fit()
                resid_std = np.std(res.resid)
                resids.append((p, resid_std))
            except:
                continue
        best_period = min(resids, key=lambda x: x[1])[0]
        sub_titles.append(f'Seasonal Component (detected period: {best_period}{period})')

    # check if it is add or multi model by residuals
    decomp_add = sm_seasonal.seasonal_decompose(df[vfield], period=seasonal_period, model='additive')
    resid_add = decomp_add.resid.dropna()
    eval_add = dict(std=np.std(resid_add),
                    mse=mean_squared_error([0]*len(resid_add), resid_add),
                    adf_p=adfuller(resid_add)[1])

    try:
        # Multiplicative seasonality is not appropriate for zero and negative values
        decomp_mul = sm_seasonal.seasonal_decompose(df[vfield], period=seasonal_period, model='multiplicative')
        resid_mul = decomp_mul.resid.dropna()
        eval_mul = dict(std=np.std(resid_mul),
                        mse=mean_squared_error([0] * len(resid_mul), resid_mul),
                        adf_p=adfuller(resid_mul)[1])
        mode = 'additive'
        if eval_mul['std'] < eval_add['std'] and eval_mul['mse'] < eval_add['mse']:
            mode = 'multiplicative'
        sub_titles.append(f'Residual Component ({mode} model)')
    except:
        sub_titles.append(f'Residual Component')

    # make 3 subplots with title
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=sub_titles)
    match algo:
        case 'stl':
            # Seasonal-Trend decomposition using LOESS (additive)
            decomp = STL(df[vfield], period=seasonal_period, robust=robust).fit()
            # variance of residuals + seasonality
            resid_seas_var = (decomp.resid + decomp.seasonal).var()
            # seasonal strength
            strength = 1 - (decomp.resid.var() / resid_seas_var)
        case 'add':
            # additive
            decomp = decomp_add
        case 'multi':
            # multiplicative
            decomp = decomp_mul
        case '_':
            return fig

    fig.add_trace(go.Scatter(x=df.index, y=df[vfield], name=vfield, mode='lines'), 1, 1)
    fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend.values, name='Trend', mode='lines'), 1, 1)
    fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal.values, name='Season', mode='lines'), 2, 1)
    fig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid.values, name='Resid', mode='markers'), 3, 1)

    fig.update_yaxes(row=1, col=1, title_text=vfield)
    fig.update_yaxes(row=2, col=1, title_text=vfield)
    fig.update_yaxes(row=3, col=1, title_text=vfield)
    title = f'Decomposition by {algo}, [{df.index.min()} ~ {df.index.max()}]'
    fig.update_layout(title=title, showlegend=False, xaxis_title='', hovermode='x')
    return fig


"""
plt ts predict chart
# ols(线性普通最小二乘), lowess(局部加权线性回归), rolling(移动平均线), ewm(指数加权移动平均), expanding(扩展窗)
{"pid": "ts", "ts": "time", "field": "dena74",  "period": "MS", "agg": "mean", "algo": "ets", "trend": "add", "season": "add"}
"""
def plt_ts_predict(ts_df, cfg, fields):
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
    period=cfg.get('period')

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
def plt_ts_anomaly(df, cfg, fields):
    fig = go.Figure()
    if cfg.get('vf') is None:
        return fig

    tsf = df.index.name
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

    vf = cfg.get('vf')
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
            y_pred = [-1 if v < th_lower else 1 if v > th_upper else 0 for v in df[vf].values]
            th_lower = [th_lower] * len(y_pred)
            th_upper = [th_upper] * len(y_pred)
        case 'gaps':
            if cfg.get('cat'):
                # get all cat names with missing values
                cat_values = df[df[[vf]].isna().any(axis=1)][cfg.get('cat')].unique().astype(str)
                cat_values.sort()
                df = df[df[cfg.get('cat')].isin(cat_values)]
            y_pred = df[[vf]].isna().any(axis=1).astype(int).tolist()
        case 'diff':
            # check change Point (break point)
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
    fig = make_subplots(rows=2, cols=1, row_heights=[600, 200], shared_xaxes=True, vertical_spacing=0.02)
    # index is ts and ts field is column
    df.reset_index(inplace=True)

    if cfg.get('cat'):
        cf = cfg['cat']
        # sort by name
        cat_name = df[cf].unique().astype(str)
        # add original data lines
        for name in cat_name:
            line_df = df[df[cf] == name]
            fig.add_trace(go.Scatter(x=line_df[tsf], y=line_df[vf], name=name), 1, 1)
        fig.update_yaxes(row=1, col=1, title_text=f'{vf} ({cfg.get("agg")})')
        fig.update_layout(legend_title_text=f'{cf} ({len(cat_name)})')
    else:
        fig.add_trace(go.Scatter(x=df[tsf], y=df[vf], name=vf, showlegend=False), 1, 1)
        fig.update_yaxes(row=1, col=1, title_text=f'{vf} ({cfg.get("agg")})')
        if method in ['quantile', 'zscore']:
            # add mean line as reference
            fig.add_hline(y=df[vf].mean(), row=1, col=1, line_color='gray', annotation_text="mean")

    total_outliers = 0
    # outlier markers
    if len(y_pred) > 0:
        # add outlier column to df
        df['outlier'] = y_pred
        # filter outliers
        outlier_df = df[df['outlier'] != 0]
        total_outliers = len(outlier_df)
        if method != 'gaps':
            # add marker to original ts chart
            fig.add_trace(go.Scatter(x=outlier_df[tsf], y=outlier_df[vf], name='outlier', mode='markers',
                                     showlegend=False, marker=dict(color='red')), 1, 1)

        outlier_df = outlier_df[[tsf, 'outlier']]
        pos_df = outlier_df[outlier_df['outlier'] > 0]
        pos_df = pos_df.groupby(tsf).sum()
        neg_df = outlier_df[outlier_df['outlier'] < 0]
        neg_df = neg_df.groupby(tsf).sum()
        alert_df = pd.concat([pos_df, neg_df])
        if len(alert_df) > 0:
            alert_df.reset_index(inplace=True, drop=False)
            # outliers with binary index (0, 1 or -1)
            fig.add_trace(go.Scatter(x=alert_df[tsf], y=alert_df['outlier'], name='alert', mode='markers',
                                     showlegend=False, marker=dict(color='red')), 2, 1)
            [fig.add_shape(type="line", x0=alert_df[tsf].loc[i], y0=0, x1=alert_df[tsf].loc[i],
                           y1=alert_df['outlier'].loc[i], line_color='gray', row=2, col=1) for i in alert_df.index]
            fig.add_hline(y=0, row=2, col=1, line_dash='dash', line_color='gray')



    # disable it for performance when there are too many lines
    if method in ['quantile_XXX', 'zscore_XXX', 'diff_XXX', 'rolling_XXX']:
        # add tolerance area
        fig.add_trace(go.Scatter(x=df[tsf], y=th_lower, name='lower', showlegend=False,
                                 line=dict(color='rgba(255,228,181,0.2)', width=0, dash='dot')), 1, 1)
        fig.add_trace(go.Scatter(x=df[tsf], y=th_upper, name='upper', showlegend=False, fill='tonexty',
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

    fig.update_yaxes(row=2, col=1, title=dict(text='outlier count'))
    fig.update_layout(title=f'Outliers: {total_outliers}/{len(y_pred)}, Alog: {method}')
    return fig

"""
plt ts similarity detection chart
{"pid": "ts", "ts": "time", "field": "dena74",  "method": "kmeans"}
"""
def plt_ts_similarity(df, cfg, fields):
    fig = go.Figure()
    if cfg.get('cat') is None or cfg.get('period') is None:
        return fig

    ts_name=df.index.name
    method = cfg.get('method', 'kmeans')
    metric = cfg.get('metric', 'euclidean')
    threshold = cfg.get('threshold')
    clusters = cfg.get('clusters', 2)

    # vf is list and cf is string
    vf = cfg.get('vf')
    cf = cfg.get('cat')

    # shape (n_samples, n_features, n_timesteps)
    d3_data = []
    d3_labels = []
    for name, group in df.groupby(cf):
        ffv = group[vf].T.values
        d3_data.append(ffv)
        d3_labels.append(name)
    X_np = np.stack(d3_data)

    cluster_ids = None
    match method:
        case 'kmeans':
            # shape (n_samples, n_features, n_timesteps) = [n_instances, n_dimensions, series_length]
            # n_samples = n_instances = n_categories
            # 单变量ex: 1个数值域特征，每天24个时间点，共5天的数据->(5,1,24)，按时间天聚类
            # 多变量ex: 1个类别域有3个类别值，每个类别6个数值特征，每个特征有24个时间点->(3,6,24)，按类别聚类
            # X_np = number_df.values.T.reshape(len(num_fields), 1, len(df))
            clst = TimeSeriesKMeans(n_clusters=clusters, metric='euclidean', max_iter=500)
            clst.fit(X_np)
            cluster_ids = clst.labels_
            cluster_cnt = dict(Counter(cluster_ids))
            cluster_dict = {}
            for idx, cluster in enumerate(cluster_ids):
                cluster_dict[d3_labels[idx]] = cluster
            cluster_center = clst.cluster_centers_
            silhouette_avg = silhouette_score(X_np.reshape(X_np.shape[0], -1), clst.labels_)
        case 'dbscan':
            # Density-Based Spatial Clustering of Applications with Noise，具有噪声的基于密度的聚类方法(cluster-based)
            #  ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’,
            #  ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
            #  ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’
            # 'minkowski' does not work
            distance = threshold if threshold else 0.5
            clst = DBSCAN(eps=distance, metric=metric)
            clst.fit(df[[vf]])
            cluster_ids = clst.labels_
        case '_':
            return fig

    title = f'Clustering method: {method}, Clusters: {clusters}, Silhouette: {silhouette_avg:.2f}'
    # sorted category names
    cat_name = df[cf].unique().astype(str)
    df['cluster'] = df[cf].map(cluster_dict)
    if clusters < 4:
        col_wrap = 1
    else:
        col_wrap = 2


    if len(vf) == 1:
        df['center'] = None
        # show lines by clusters
        fig = px.line(df, x=df.index, y=vf[0], color=cf, facet_col='cluster', facet_col_wrap=col_wrap)
        # show cluster center for single value field
        for k, v in cluster_dict.items():
            # add data to df
            df.loc[df[cf].astype(str) == k, 'center'] = cluster_center[v][0].tolist()
        # build center lines
        lines = px.line(df, x=df.index, y='center', facet_col='cluster', facet_col_wrap=col_wrap,
                    color_discrete_sequence=['gray'], line_dash_sequence=['dot'])
        fig.add_traces(lines.data)
    elif cfg.get('d2'):
        df['d0'] = None
        df['d1'] = None
        pca = PCA(n_components=2)
        for k, v in cluster_dict.items():
            pca_data = pca.fit_transform(df[df[cf].astype(str) == k][vf])
            df.loc[df[cf].astype(str) == k, 'd0'] = pca_data[:,0].flatten().tolist()
            df.loc[df[cf].astype(str) == k, 'd1'] = pca_data[:,1].flatten().tolist()
        # convert every time series to a data point by category
        cluster_cat_df = df.groupby(['cluster', cf])[['d0', 'd1']].mean().reset_index()
        # delete NaN
        cluster_cat_df.dropna(inplace=True)
        # convert cluster field to string for discrete color of plotly
        cluster_cat_df['cluster'] = cluster_cat_df['cluster'].astype(str)
        # show 2d scatter plot and same cluster has same color
        fig = px.scatter(cluster_cat_df, x='d0', y='d1', color='cluster', symbol=cf)
    else:
        df['center'] = None
        pca = PCA(n_components=1)
        for k, v in cluster_dict.items():
            pca_data = pca.fit_transform(df[df[cf].astype(str) == k][vf])
            df.loc[df[cf].astype(str) == k, 'center'] = pca_data.flatten().tolist()
        # build center lines
        fig = px.line(df, x=df.index, y='center', color=cf, facet_col='cluster', facet_col_wrap=col_wrap)
    fig.for_each_annotation(lambda a: a.update(text=cluster_cnt[int(a.text.split("=")[-1])]))
    fig.for_each_xaxis(lambda x: x.update(title=''))
    fig.update_layout(title=title, legend_title_text=f'{cf} ({len(cat_name)})', xaxis_title='', hovermode='x')
    return fig


"""
plt ts active noice reduction chart
{"pid": "ts", "ts": "time", "field": "dena74",  "method": "zscore"}
"""
def plt_ts_anc(df, cfg, fields):
    fig = go.Figure()
    if cfg.get('vf') is None:
        return fig

    method = 'quantile'
    if cfg.get('method'):
        method = cfg['method']

    # contamination: 默认为0.05，即5%的异常值
    # irq_coff: 默认为1.6，即1.6倍IQR
    # sigma_coff: 默认为3，即3倍标准差
    # use same parameter name for all methods
    threshold = None
    if cfg.get('threshold'):
        threshold = cfg['threshold']

    vf = cfg['vf']
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
            # get top 5 by power
            top_power = psd_df.nlargest(5, 'power')
            top_power = top_power[top_power['power'] > 0]
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
    fig.update_xaxes(row=2, col=1, title=f'Frequency (1/{cfg.get("period")})')
    fig.update_yaxes(row=2, col=1, title=f'{vf} (Power/Hz)')
    fig.update_yaxes(row=1, col=1, title=f'{vf} ({cfg.get("agg")})')
    fig.update_layout(hovermode='x')
    return fig


"""
resample time series based on config 'period' and 'agg'
"""
def ts_resample(df: pd.DataFrame, cfg: dict, fields: list, trans: dict = None, fill: bool = True):
    ts_name = df.index.name
    cat_field = cfg.get('cat')
    if cfg.get('period') is None and cfg.get('agg'):
        # convert ts index to a column for groupby
        df.reset_index(inplace=True)
        # group by ts and cat
        group_fields = [ts_name, cat_field] if cat_field else [ts_name]
        # aggregate data
        df = df.groupby(group_fields).agg(cfg['agg'])
        # convert multiple index to columns
        df.reset_index(inplace=True)
        if cat_field:
            df.sort_values(by=group_fields, key=lambda x: x.astype(str) if x.name == cat_field else x,
                                 inplace=True)
        # set ts field as index
        df.set_index(ts_name, inplace=True)
        # sort timestamp
        df.sort_index(inplace=True)
        return df, None

    period = cfg.get('period')
    period_def = dict(s=0, min=1, h=2, D=3, W=4, MS=5, QS=6, YS=7)
    # aggregated by period (YS,QS,MS,W,D,h,min,s)
    # agg: sum, mean, median, min, max, std, count
    agg = cfg.get('agg', 'mean')

    ts_field = [field for field in fields if field.get('name') == ts_name][0]
    # find config of ts field to confirm period
    if ts_field.get('name') == ts_name:
        gap_min = 1
        interval_min = 1
        if ts_field.get('gap'):
            # minimal gap between two points
            gap_min = int(ts_field['gap'])
        if ts_field.get('resample'):
            # user defined minimal interval in dataset
            interval_min = int(ts_field['resample'])

        # get max resample period from the two values (unit: minute)
        resample_period = max(gap_min, interval_min)
        if resample_period < 60 and period_def[period] < 2:
            # update period to max minute
            period = f'{resample_period}T'
        elif resample_period > 60 and resample_period < 1440 and period_def[period] < 3:
            # update period to max hour
            period_h = math.ceil(resample_period / 60)
            period = f'{period_h}H'
        elif resample_period > 1440 and resample_period < 10080 and period_def[period] < 4:
            # update period to max hour
            period_day = math.ceil(resample_period / 1440)
            period = f'{period_day}D'
        elif resample_period > 10080 and resample_period < 43200 and period_def[period] < 5:
            # update period to max hour
            period_wk = math.ceil(resample_period / 10080)
            period = f'{period_wk}W'


    # create a full time index
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=period)
    # both config['vf'] and config['cat'] are list
    if cfg.get('cat'):
        cat_field = cfg['cat']
        resampled_df = df.groupby(cat_field).apply(lambda g: g.resample(period).agg(agg).reindex(full_range))
        # set ts_name to index name of None
        resampled_df.index.names = [cat_field, ts_name]
        # convert multiple index to regular columns for sorting
        resampled_df.reset_index(inplace=True)
        # sort  by ts_name and cat_field following time and dictionary order
        resampled_df.sort_values(by=[ts_name, cat_field], key=lambda x: x.astype(str) if x.name == cat_field else x, inplace=True)
        # set ts_name as index
        resampled_df.set_index(ts_name, inplace=True)
    else:
        # all are value fields
        resampled_df = df.resample(period).agg(agg).reindex(full_range)

    resampled_df.index.name = ts_name
    if fill:
        # value fields
        num_fields = [field for field in fields if field['attr'] in ['conti', 'disc']]
        # handle missing value
        for field in num_fields:
            vf = field['name']
            if vf in resampled_df.columns and resampled_df[vf].isnull().any():
                # this field has null value after resampling
                if field.get('miss'):
                    # high priority is field config
                    miss_operation = field['miss']
                elif trans and trans.get('miss'):
                    miss_operation = trans['miss']
                else:
                    continue

                match miss_operation:
                    case 'mean':
                        resampled_df[vf] = resampled_df[vf].fillna(resampled_df[vf].mean())
                    case 'median':
                        resampled_df[vf] = resampled_df[vf].fillna(resampled_df[vf].median())
                    case 'mode':
                        resampled_df[vf] = resampled_df[vf].fillna(resampled_df[vf].mode())
                    case 'prev':
                        resampled_df[vf] = resampled_df[vf].ffill()
                        resampled_df[vf] = resampled_df[vf].bfill()
                    case 'next':
                        resampled_df[vf] = resampled_df[vf].bfill()
                        resampled_df[vf] = resampled_df[vf].ffill()
                    case 'zero':
                        resampled_df[vf] = resampled_df[vf].fillna(value=0)
    resampled_df.sort_index(inplace=True)
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