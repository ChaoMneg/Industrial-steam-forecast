import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns

# modelling
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score,cross_val_predict,KFold
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler,StandardScaler


# 导入数据
with open("F:\data\zhengqi_train.txt")  as fr:
    data_train=pd.read_table(fr,sep="\t")
with open("F:\data\zhengqi_test.txt") as fr_test:
    data_test=pd.read_table(fr_test,sep="\t")

# 合并测试集和训练集
data_train["oringin"]="train"
data_test["oringin"]="test"
data_all=pd.concat([data_train,data_test],axis=0,ignore_index=True)
# 查看前5行数据
# print(data_all.head())

# # 查看特征分布
# fig = plt.subplots(figsize=(30,20))
# j = 1
# for column in data_all.columns[0:-2]:
#     plt.subplot(5, 8, j)
#     g = sns.kdeplot(data_all[column][(data_all["oringin"] == "train")], color="Red", shade = True)
#     g = sns.kdeplot(data_all[column][(data_all["oringin"] == "test")], ax =g, color="Blue", shade= True)
#     g.set_xlabel(column)
#     g.set_ylabel("Frequency")
#     g = g.legend(["train","test"])
#     j += 1
# plt.show()

# 删除特征"V5","V9","V11","V17","V22","V28"，训练集和测试集分布不均
data_all.drop(["V5","V9","V11","V17","V22","V28"],axis=1,inplace=True)
data_train1 = data_all[data_all["oringin"] == "train"].drop("oringin", axis=1) # 提取训练集数据,已删除部分特征

"""---------------------------------------------------------------------------"""
# # 特征参数--没发现作用(修改过，图的效果暂时不好)
# fcols = 2
# frows = len(data_train.columns) # 40
# # plt.figure(figsize=(5 * fcols, 4 * frows)) # 设置图像的大小
# plt.figure(figsize=(20, 30)) # 设置图像的大小
# i = 0
# for col in data_train1.columns:
#     i += 1
#     ax = plt.subplot(5, 8, i)
#     sns.regplot(x=col, y='target', data=data_train, ax=ax,
#                 scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
#                 line_kws={'color': 'k'})
#
#     sns.distplot(data_train[col].dropna(), fit=stats.norm)
#     plt.xlabel(col)
#     plt.ylabel('target')
# plt.show()
"""------------------------------------------------------------------------------"""
# # 找出相关程度，绘制热力图
# plt.figure(figsize=(20, 16))  # 指定绘图对象宽度和高度
# colnm = data_train1.columns.tolist()  # 列表头
# mcorr = data_train1[colnm].corr(method="spearman")  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
# mask = np.zeros_like(mcorr, dtype=np.bool)  # 构造与mcorr同维数矩阵 为bool型
# mask[np.triu_indices_from(mask)] = True  # 遮挡热力图上三角部分的mask
# cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
# g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
# plt.show()

# 移除相关变量的阈值
threshold = 0.1
# 绝对值相关矩阵
corr_matrix = data_train1.corr().abs() # 相关系数矩阵加绝怼值 33*33
# 删除与target的相关系数小于threshold的特征
drop_col=corr_matrix[corr_matrix["target"]<threshold].index
data_all.drop(drop_col,axis=1,inplace=True)

# 数据基本统计量，包含：
# count：数量  mean：均值  std：标准差  min：最小值  25%：下四分位  50%：中位数  75%：上四分位  max：最大值
# cols_numeric=list(data_all.columns)
# cols_numeric.remove("oringin")
# def scale_minmax(col):
#     return (col-col.min())/(col.max()-col.min())
# scale_cols = [col for col in cols_numeric if col!='target']
# data_all[scale_cols] = data_all[scale_cols].apply(scale_minmax,axis=0)
# data_all[scale_cols].describe()
# print(data_all[scale_cols].describe())

# # 检查Box-Cox变换对连续变量分布的影响
# fcols = 6
# frows = len(cols_numeric) - 1
# plt.figure(figsize=(4 * fcols, 4 * frows))
# i = 0
#
# for var in cols_numeric:
#     if var != 'target': # 排除target列
#         dat = data_all[[var, 'target']].dropna() # 单个特征列+target列，dropna()过滤缺失值(相当于过滤掉拼接的测试集)
#         # 绘制原始数据图像，；连续正太分布
#         i += 1
#         plt.subplot(frows, fcols, i)
#         sns.distplot(dat[var], fit=stats.norm)
#         plt.title(var + ' Original')
#         plt.xlabel('')
#
#         # 计算偏度
#         """
#         偏度是统计数据分布非对称程度的数字特征
#         对于正态分布数据，偏度应约为0
#         对于单峰连续分布，偏度值> 0意味着分布的右尾部有更多权重
#         在统计上，函数skewtest可用于确定偏度值是否足够接近0
#         """
#         i += 1
#         plt.subplot(frows, fcols, i)
#         _ = stats.probplot(dat[var], plot=plt)
#         plt.title('skew=' + '{:.4f}'.format(stats.skew(dat[var])))
#         plt.xlabel('')
#         plt.ylabel('')
#
#         # 相关系数
#         i += 1
#         plt.subplot(frows, fcols, i)
#         plt.plot(dat[var], dat['target'], '.', alpha=0.5)
#         plt.title('corr=' + '{:.2f}'.format(np.corrcoef(dat[var], dat['target'])[0][1])) # corrcoef()计算相关系数函数
#
#
#         # Box-Cox变换
#         """
#         广义幂变换方法，用于连续的响应变量不满足正态分布的情况
#         Box-Cox变换之后，可以一定程度上减小不可观测的误差和预测变量的相关性
#         Box-Cox变换的主要特点是引入一个参数，通过数据本身估计该参数进而确定应采取的数据变换形式，Box-Cox变换可以明显地改善数据的正态性、对称性和方差相等性
#         """
#         i += 1
#         plt.subplot(frows, fcols, i)
#         trans_var, lambda_var = stats.boxcox(dat[var].dropna() + 1)
#         trans_var = scale_minmax(trans_var)
#         sns.distplot(trans_var, fit=stats.norm)
#         plt.title(var + ' Tramsformed')
#         plt.xlabel('')
#
#         # 计算变换后的偏度
#         i += 1
#         plt.subplot(frows, fcols, i)
#         _ = stats.probplot(trans_var, plot=plt)
#         plt.title('skew=' + '{:.4f}'.format(stats.skew(trans_var)))
#         plt.xlabel('')
#         plt.ylabel('')
#
#         # 计算变换后的相关系数
#         i += 1
#         plt.subplot(frows, fcols, i)
#         plt.plot(trans_var, dat['target'], '.', alpha=0.5)
#         plt.title('corr=' + '{:.2f}'.format(np.corrcoef(trans_var, dat['target'])[0][1]))
# plt.show()

"""---------------------------------------------------------------------------"""
# 对特征进行Box-Cox变换---添加后报错，注释后不影响最终预测结果
# cols_transform=data_all.columns[0:-2] # 取所有特征的名字
# for col in cols_transform:
#     # 对所有列进行变换
#     data_all.loc[:,col], _ = stats.boxcox(data_all.loc[:,col]+1)  # loc:按索引值定位
"""---------------------------------------------------------------------------"""
# # -------没看懂--------------
# # print(data_all.target.describe())
#
# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# sns.distplot(data_all.target.dropna() , fit=stats.norm)
# plt.subplot(1,2,2)
# _=stats.probplot(data_all.target.dropna(), plot=plt) # probplot()-根据指定理论分布的分位数（默认情况下的正态分布）生成样本数据的概率图
# # plt.show()

#Log转换因变量来提高常态---生成新标签
sp = data_train.target # 取训练集target列
data_train.target1 =np.power(1.5,sp) # 对1.5取sp次方

# # print(data_train.target1.describe())
# #
# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# sns.distplot(data_train.target1.dropna(),fit=stats.norm)
# plt.subplot(1,2,2)
# _=stats.probplot(data_train.target1.dropna(), plot=plt)
# # plt.show()

# 查看特征分布，经过boxcox函数变换+特征选择
# fig = plt.subplots(figsize=(30,20))
# j = 1
# for column in data_all.columns[0:-2]:
#     plt.subplot(5, 8, j)
#     g = sns.kdeplot(data_all[column][(data_all["oringin"] == "train")], color="Red", shade = True)
#     g = sns.kdeplot(data_all[column][(data_all["oringin"] == "test")], ax =g, color="Blue", shade= True)
#     g.set_xlabel(column)
#     g.set_ylabel("Frequency")
#     g = g.legend(["train","test"])
#     j += 1
# plt.show()

# 获取训练集
def get_training_data():
    # extract training samples
    df_train = data_all[data_all["oringin"]=="train"]
    df_train["label"]=data_train.target1 # log处理过的新标签
    # 拆分因变量和特征
    y = df_train.target
    X = df_train.drop(["oringin","target","label"],axis=1)
    X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.3,random_state=100) # 交叉验证函数，随机划分训练集和验证集
    return X_train,X_valid,y_train,y_valid

# 获取测试集（无标签）
def get_test_data():
    df_test = data_all[data_all["oringin"]=="test"].reset_index(drop=True)
    return df_test.drop(["oringin","target"],axis=1)


# 均方根误差
def rmse(y_true, y_pred):
    diff = y_pred - y_true  # 预测-真实
    sum_sq = sum(diff ** 2)
    n = len(y_pred)
    return np.sqrt(sum_sq / n)

# 均方误差---可以直接输出，没必要写
def mse(y_ture, y_pred):
    return mean_squared_error(y_ture, y_pred)

# sklearn库中make_scorer()函数进行评分
rmse_scorer = make_scorer(rmse, greater_is_better=False) # False，返回的为score的负值，值越低越好
mse_scorer = make_scorer(mse, greater_is_better=False)


# 根据模型检查离群点
def find_outliers(model, X, y, sigma=3):
    # 预测y值
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # 如果预测失败，尝试拟合模型
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    # 计算预测值与实际值的残差（残差在数理统计中是指实际观察值与估计值（拟合值）之间的差）
    resid = y - y_pred
    mean_resid = resid.mean() # 求平均值
    std_resid = resid.std() # 求标准差

    z = (resid - mean_resid) / std_resid # 残差的正态性检验
    outliers = z[abs(z) > sigma].index # 异常值

    # 结果
    print('R2=', model.score(X, y))
    print('rmse=', rmse(y, y_pred))
    print("mse=", mean_squared_error(y, y_pred))
    print('---------------------------------------')

    print('mean of residuals:', mean_resid)  # 残差均值
    print('std of residuals:', std_resid) # 残差标准差
    print('---------------------------------------')

    print(len(outliers), 'outliers:') # 异常值
    print(outliers.tolist()) # 将outliers转换为list

    # 绘图部分
    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred');

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred')

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('z')

    plt.savefig('outliers.png')

    return outliers

X_train, X_valid,y_train,y_valid = get_training_data() # 获取训练+验证集
test=get_test_data() # 测试集

# 使用岭回归发现和删除离群点
# Ridge 回归又称岭回归，它是普通线性回归加上 L2 正则项，用来防止训练过程中出现的过拟合
outliers = find_outliers(Ridge(), X_train, y_train)

# 删除异常值
X_outliers=X_train.loc[outliers]
y_outliers=y_train.loc[outliers]
X_t=X_train.drop(outliers)
y_t=y_train.drop(outliers)

# 获取剔除异常值后的训练集
def get_trainning_data_omitoutliers():
    y1=y_t.copy()
    X1=X_t.copy()
    return X1,y1


# 训练模型
def train_model(model, param_grid=[], X=[], y=[], splits=5, repeats=5):
    # 获取训练集
    if len(y) == 0:
        X, y = get_trainning_data_omitoutliers()

    # 交叉验证函数
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats) # p次k折交叉验证

    # 如果param_grid给出则进行网格搜索
    if len(param_grid) > 0:
        # 设置搜索参数
        # GridSearchCV()在指定的范围内自动搜索具有不同超参数的不同模型组合，自动选择输入参数中的最优组合
        gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                               scoring="neg_mean_squared_error",
                               verbose=1, return_train_score=True)

        # 评估模型
        gsearch.fit(X, y)

        # 选择最佳模型
        model = gsearch.best_estimator_
        best_idx = gsearch.best_index_

        # 获取最佳模型的cv-scores
        grid_results = pd.DataFrame(gsearch.cv_results_) # 生成一个二维表
        cv_mean = abs(grid_results.loc[best_idx, 'mean_test_score'])
        cv_std = grid_results.loc[best_idx, 'std_test_score'] # loc:按索引值定位

    # 获得模型的交叉评分
    else:
        grid_results = []
        cv_results = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=rkfold)  # 交叉验证
        cv_mean = abs(np.mean(cv_results)) # 均值
        cv_std = np.std(cv_results) # 标准差

    # 将cv-score的mean、std添加到序列里
    cv_score = pd.Series({'mean': cv_mean, 'std': cv_std})

    # 使用模型预测y
    y_pred = model.predict(X)

    # 输出需要统计的数据
    print('----------------------')
    print(model) # 模型
    print('----------------------')
    print('score=', model.score(X, y)) # 得分
    print('rmse=', rmse(y, y_pred)) # 均方根误差
    print('mse=', mse(y, y_pred)) # 均方误差
    print('cross_val: mean=', cv_mean, ', std=', cv_std) # cv-scores的均值，标准差

    # 绘图---暂时未详细解读
    y_pred = pd.Series(y_pred, index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid) / std_resid
    n_outliers = sum(abs(z) > 3)

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('y_pred');
    plt.title('corr = {:.3f}'.format(np.corrcoef(y, y_pred)[0][1]))
    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('y - y_pred');
    plt.title('std resid = {:.3f}'.format(std_resid))

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    plt.xlabel('z')
    plt.title('{:.0f} samples with z>3'.format(n_outliers))

    return model, cv_score, grid_results

# 存储模型和最佳得分
opt_models = dict()
score_models = pd.DataFrame(columns=['mean','std'])

# RepeatedKFold()函数的参数，在train_model()里使用
splits=5 # 将数据集划分为5等份
repeats=5 # K-Fold重复次数

model='LinearSVR'
opt_models[model] = LinearSVR()

crange = np.arange(0.1,1.0,0.1)  # 0.1-1.0的一个list，间隔0.1
param_grid = {'C':crange,
             'max_iter':[1000]}

# 训练模型
# 入参：model, param_grid=[], X=[], y=[]
# 返回：model, cv_score, grid_results
opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid,splits=splits, repeats=repeats)

cv_score.name = model
score_models = score_models.append(cv_score)

# 绘图
plt.figure()
plt.errorbar(crange, abs(grid_results['mean_test_score']),abs(grid_results['std_test_score'])/np.sqrt(splits*repeats))
plt.xlabel('C')
plt.ylabel('score')
plt.show()
