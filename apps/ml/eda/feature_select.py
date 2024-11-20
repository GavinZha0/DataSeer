import numpy as np
import pandas as pd
# import shap
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import RFECV, GenericUnivariateSelect, SelectFromModel
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression
import xgboost as xgb
import featuretools as ftool
from sklearn import preprocessing as pp
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


"""
get feature score or importance
单变量特征选择(Univariate feature selection)是通过基于单变量统计检验来选择最优特征实现的。
SelectPercentile: 只保留用户指定百分比的最高得分的特征；
SelectKBest: 只保留 k 个最高分的特征；
SelectFpr: False Positive Rate, 保留特征的最高p值
SelectFdr: False Discovery Rate, 保留特征的最高p值
GenericUnivariateSelect: 通用单变量特征选择器(方差分析,ANOVA),通过结构化策略进行特征选择或者通过超参数搜索估计器进行特征选择。
score_func： f_classif, mutual_info_classif(类别vs类别) and chi2(类别vs类别) for classification; 
            f_regression and mutual_info_regression for reg
mode: percentile(选取排名前x%的特征), k_best(依据相关性,选择k个最强大的特征), fpr(误报率,假阳性率), 
        fdr(基于错误发现率), fwe(多重比较谬误率,基于族系误差)
the 'percentile' and 'kbest' modes are supporting unsupervised feature selection (when y is None).
"""
def feature_corr_filter(X: pd.DataFrame, y: pd.DataFrame, config):
    scores = []
    method = 'f_test'
    if config.get('method'):
        method = config['method']

    mode = 'percentile'
    param = 20
    if config.get('mode'):
        mode = config['mode']
        if mode == 'percentile':
            # 保留排名前10%的特征
            param = 20
        elif mode == 'k_best':
            # 保留全部特征
            param = 'all'
        else:
            # FPR, False Positive Rate, 假阳性率, 是指被我们预测为正但实际为负的样本的比例
            # FDR, False Discovery Rate, 错误发现率, 在我们拒绝原假设的样本中，有多大比例是犯了一类错误的
            # FWER, family-wise error rate, 族系误差率, 指至少出现一次一类错误的概率
            # 保留FPR, FDR或FWER小于5%的特征
            param = 0.05

    classif = False
    y_array = []
    if y is not None:
        # ONE target only
        y_array = y.iloc[:, 0]
        if 'category' in y.dtypes.to_list():
            classif = True

    match method:
        case 'f_test':
            # ANOVA, 多变量，连续型VS类别型
            func = f_regression
            if classif:
                func = f_classif
            selector = GenericUnivariateSelect(score_func=func, mode=mode, param=param).fit(X, y_array)
            prime = selector.get_support()
            # score1 = -np.log10(selector.pvalues_)
            scores = selector.scores_
        case 'm_info':
            # 互信息方法可以捕捉任何一种统计依赖，如遇系数矩阵，推荐此法
            # 多变量，类别型VS类别型
            func = mutual_info_regression
            if classif:
                func = mutual_info_classif
            selector = GenericUnivariateSelect(score_func=func, mode=mode, param=param).fit(X, y_array)
            # function get_support is not available when mode is fpr, fdr and fwe
            prime = [True] * len(X.columns)
            if selector.pvalues_ is not None:
                prime = selector.get_support()
            scores = selector.scores_
        case 'chi2':
            # 卡方分布是通过统计非负特征和目标变量的卡方统计量来分析两者关系来衡量特征的重要程度
            # 多变量，类别型VS类别型
            tp_X = pp.MinMaxScaler().fit_transform(X)
            selector = GenericUnivariateSelect(score_func=chi2, mode=mode, param=param).fit(tp_X, y_array)
            prime = selector.get_support()
            scores = selector.scores_

    # prime: Ture or False
    f_score = pd.DataFrame({'name': X.columns.to_list(), 'prime': prime, 'value': scores.round(3)})
    f_score.sort_values(by='value', ascending=False, inplace=True)
    resp = f_score.to_dict(orient='list')
    resp['method'] = config.get('method')
    return resp


"""
get feature score or importance
SelectFromModel: 基于模型评分
SelectFromModel 是一个 meta-transformer（元转换器） ，它可以用来处理任何带有 coef_ 或者 feature_importances_ 属性的训练之后的评估器。 
如果相关的coef_ 或者 featureimportances 属性值低于预先设置的阈值threshold，这些特征将会被认为不重要并且移除掉.
当threshold为None，如果估计器的参数惩罚设置为l1（Lasso），则使用的threshold默认为1e-5。否则，默认使用mean。
SHAP: Shapley Additive Explanations, 可解释性，基于边际贡献,通过对特征重要性进行评估，解释模型对每个特征的贡献。
"""
def feature_model_eval(X: pd.DataFrame, y: pd.DataFrame, config):
    scores = []
    model = 'linear'
    if config.get('model'):
        model = config['model']

    classif = False
    if y is not None:
        # ONE target only
        yy = y.iloc[:, 0]
        if 'category' in y.dtypes.to_list():
            classif = True

    match model:
        case 'linear':
            if classif:
                # LogisticRegression
                selector = SelectFromModel(LogisticRegressionCV(penalty='l2', solver='lbfgs'), threshold='mean').fit(X, yy)
                prime = selector.get_support()
                scores = np.abs(selector.estimator_.coef_[0])
            else:
                selector = SelectFromModel(LinearRegression(), threshold='mean').fit(X, yy)
                prime = selector.get_support()
                scores = selector.estimator_.coef_
        case 'lasso':
            # L1正则化，又叫Lasso Regression
            # L2正则化，又叫Ridge Regression
            # L1正则化将系数l1范数作为惩罚项添加损失函数上，这就迫使那些弱的特征所对应的系数变成0
            # 参数C控制稀疏性：C越小，被选中的特征越少
            if classif:
                # LinearSVC基于liblinear库实现,svm.SVC基于libsvm库实现,数据量大时LinearSVC占优
                selector = SelectFromModel(svm.LinearSVC(C=0.01, penalty='l1', dual=False), threshold='mean').fit(X, yy)
                prime = selector.get_support()
                scores = np.abs(selector.estimator_.coef_[0])
            else:
                # 参数alpha越大，被选中的特征越少
                # LassoCV, 通过交叉验证自动确定alpha值
                selector = SelectFromModel(LassoCV(), threshold='mean').fit(X, yy)
                prime = selector.get_support()
                scores = selector.estimator_.coef_
        case 'randomf':  # RandomForestClassifier
            if classif:
                selector = SelectFromModel(RandomForestClassifier(), threshold='mean').fit(X, yy)
                prime = selector.get_support()
                scores = selector.estimator_.feature_importances_
            else:
                selector = SelectFromModel(RandomForestRegressor(), threshold='mean').fit(X, yy)
                prime = selector.get_support()
                scores = selector.estimator_.feature_importances_
        case 'xgboost':  # XGBoost
            # eXtreme Gradient Boosting
            if classif:
                selector = SelectFromModel(xgb.XGBClassifier(), threshold='mean').fit(X, yy)
                prime = selector.get_support()
                scores = selector.estimator_.feature_importances_
            else:
                selector = SelectFromModel(xgb.XGBRegressor(), threshold='mean').fit(X, yy)
                prime = selector.get_support()
                scores = selector.estimator_.feature_importances_

    # prime: Ture or False
    f_score = pd.DataFrame({'name': X.columns.to_list(), 'prime': prime, 'value': scores.round(3)})
    f_score.sort_values(by='value', ascending=False, inplace=True)
    resp = f_score.to_dict(orient='list')
    resp['method'] = config.get('method')
    return resp


"""
search feature score or importance
RFE: Recursive Feature Elimination
permutation importance（排列重要性）
null importance（排列重要性）
SHAP: Shapley Additive Explanations, 可解释性，基于边际贡献,通过对特征重要性进行评估，解释模型对每个特征的贡献。
"""
def feature_iter_search(X: pd.DataFrame, y: pd.DataFrame, config):
    scores = []
    method = 'rfe'
    if config.get('method'):
        method = config['method']

    classif = False
    yy = []
    if y is not None:
        # ONE target only
        yy = y.iloc[:, 0]
        if 'category' in y.dtypes.to_list():
            classif = True

    match method:
        case 'rfe':
            # RFE: Recursive Feature Elimination
            # 递归特征消除(RFE)的目标是通过递归考虑越来越小的特征集来选择特征。
            # soring: accuracy, recall, f1, mse,
            # Extra-Trees (Extremely randomized trees，极度随机树)
            # 相比于随机森林，极度随机表现在对决策树节点的划分上
            scoring = 'accuracy'
            if config.get('scoring'):
                scoring = config['scoring']
            if classif:
                rfe = RFECV(ExtraTreesClassifier(), min_features_to_select=2, cv=2,
                            scoring=scoring, n_jobs=1).fit(X, yy)
                prime = rfe.get_support()
                rank_max = max(rfe.ranking_) + 1
                scores = rank_max - rfe.ranking_
            else:
                rfe = RFECV(ExtraTreesRegressor(), min_features_to_select=2, cv=2,
                            scoring=scoring, n_jobs=1).fit(X, yy)
                prime = rfe.get_support()
                rank_max = max(rfe.ranking_) + 1
                scores = rank_max - rfe.ranking_
        case 'permute':
            # permutation importance（排列重要性）
            # 计算量随着特征的增加而线性增加，对于维度很高的数据基本上难以使用
            # # Histogram-Based Gradient Boosting,基于直方图的梯度提升树(受LightGBM启发)
            # HGB特征处理不需要one-hot，而是序列化即可;HGB支持缺失值，意味着不需要做missing value的处理。
            # HGB非常快，内存效率高，适用于大型数据集
            if classif:
                md = HistGradientBoostingClassifier().fit(X, yy)
                selector = permutation_importance(md, X, yy)
                prime = [True] * len(X.columns)
                scores = selector.importances_mean
            else:
                md = HistGradientBoostingRegressor().fit(X, yy)
                selector = permutation_importance(md, X, yy)
                prime = [True] * len(X.columns)
                scores = selector.importances_mean
        case 'nullimp':
            # null importance（排列重要性）
            # model: lightGBM
            prime = [True] * len(X.columns)
            scores = exe_null_imp(X, yy, classif, 10)
        case 'shap':
            shap = 888
    # prime: Ture or False
    f_score = pd.DataFrame({'name': X.columns.to_list(), 'prime': prime, 'value': scores.round(3)})
    f_score.sort_values(by='value', ascending=False, inplace=True)
    resp = f_score.to_dict(orient='list')
    resp['method'] = config.get('method')
    return resp


"""
auto detect/generate/encode features
dfs: Deep Feature Synthesis (深度特征合成)

"""
def feature_auto_detect(X: pd.DataFrame, y: pd.DataFrame, config):
    match config['method']:
        case 'dfs':
            # Deep Feature Synthesis
            es = ftool.EntitySet(id='eSet')
            es = es.add_dataframe(dataframe_name='f_engg', dataframe=X, make_index=True, index='uu_id')
            f_matrix, f_names = ftool.dfs(entityset=es, target_dataframe_name='f_engg', max_depth=2)
        case 'tsfresh':
            # TSFRESH automatically extracts 100s of features from time series.
            # TSFresh 自动从时间序列中提取 100 个特征。 这些特征描述了时间序列的基本特征，
            # 例如峰值数量、平均值或最大值或更复杂的特征，例如时间反转对称统计量。
            aaa = 666
            # extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
        case 'featurewiz':
            # https://github.com/AutoViML/featurewiz
            bbb = 666
        case 'pycaret':
            # https://github.com/pycaret/pycaret
            # PyCaret 不是一个专用的自动化特征工程库，但它包含自动生成特征的功能。
            ccc = 888

    resp = f_matrix.head(5).T.to_dict(orient='split')
    resp['method'] = config.get('method')
    return resp


## 获取lightgbm的特征重要度，这里应用的lightgbm的随机森林算法和论文内的算法逻辑保持一致
def get_feature_imp(data, y, classif=True, shuffle=False):
    if shuffle:
        y = y.copy().sample(frac=1.0)

    clf_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': None,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1
    }

    reg_params = {
        'objective': 'regression',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': None,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1
    }

    imp_df = pd.DataFrame()
    # Fit LightGBM in RF mode, it's quicker than sklearn RandomForest
    # dtrain = lgb.Dataset(data, y, free_raw_data=False)
    # Fit the model
    if classif:
        # md = lgb.train(params=clf_params, train_set=dtrain, num_boost_round=200)
        md = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.01).fit(data, y)
        imp_df['trn_score'] = roc_auc_score(y, md.predict_proba(data), multi_class='ovr')
    else:
        # md = lgb.train(params=reg_params, train_set=dtrain, num_boost_round=200)
        md = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.01).fit(data, y)
        imp_df['trn_score'] = roc_auc_score(y, md.predict_proba(data))

    # Get feature importances
    imp_df["feature"] = data.columns.to_list()
    imp_df["gain"] = md.feature_importances_
    # imp_df["importance_split"] = md.feature_importance(importance_type='split')
    return imp_df

def exe_null_imp(data, y, classif=True, nb_runs=10):
    null_imp_df = pd.DataFrame()
    dsp = ''
    actual_imp_df = get_feature_imp(data, y, classif, shuffle=False)
    # 运算Null importance
    for i in range(nb_runs):
        # Get current run importances
        imp_df = get_feature_imp(data, y, classif, shuffle=True)
        imp_df['run'] = i + 1
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'gain'].mean()
        gain_score = np.log(
            1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
        # f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        # f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        # split_score = np.log(
        #     1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
        feature_scores.append((_f, gain_score))

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'score'])
    return scores_df['score'].to_numpy()