#伊海
#utf-8
#yihaizhenhao@163.com
#3471018046@qq.com
# Python 的版本需要大于3.5
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn的版本需要大于0.20
import sklearn
assert sklearn.__version__ >= "0.20"


import numpy as np
import os

#绘图设置
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 本块代码现在无需理解，复制粘贴执行即可
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://gitee.com/yang-yizhou/dayangai/raw/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'whatever')
    opener.retrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
#报错就多运行几次
fetch_housing_data()

# 本块代码现在无需理解，复制粘贴执行即可
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
type(housing)
housing.head
housing.info()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# 本代码块的每个参数可以根据单词意思作调整，看看输出效果，以便加深理解
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet")
)
# 本块代码现在无需理解，复制粘贴执行即可
PROJECT_ROOT_DIR = "."
images_path = os.path.join(PROJECT_ROOT_DIR, "images", "housing")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://gitee.com/yang-yizhou/dayangai/raw/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/housing/" + filename
opener = urllib.request.URLopener()
opener.addheader('User-Agent', 'whatever')
opener.retrieve(url,os.path.join(images_path, filename))

# 本块代码现在无需理解，复制粘贴执行即可
import matplotlib.image as mpimg
california_img=mpimg.imread(os.path.join(images_path, filename))
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                  s=housing['population']/100, label="Population",
                  c="median_house_value", cmap=plt.get_cmap("jet"),
                  colorbar=False, alpha=0.4)

# 输出  下面这四个散点图
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
plt.show()

################
housing_target = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)
housing.info()
# 方法1
# housing.dropna(subset=["total_bedrooms"])
# 方法2
# housing.drop("total_bedrooms", axis=1)
# 方法3
median = housing["total_bedrooms"].median() #中位数
housing["total_bedrooms"].fillna(median, inplace=True)
#强烈建议探究下housing[["ocean_proximity"]]和housing["ocean_proximity"]的区别
housing_category = housing[["ocean_proximity"]]
housing_category["ocean_proximity"].value_counts()
housing_category[:10]#前十个地区的类别

from sklearn.preprocessing import OrdinalEncoder # 加载库
ordinal_encoder = OrdinalEncoder() # 创建转换器
housing_category_encoded = ordinal_encoder.fit_transform(housing_category) #转换
housing_category_encoded[:10]#前十个地区，看看效果

ordinal_encoder.categories_

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_category_onehot = cat_encoder.fit_transform(housing_category)
#housing_category_onehot和housing_category_encoded类型不一样
housing_category_onehot.toarray()[:10]

housing=housing.drop("ocean_proximity", axis=1) #先把原来的放弃
#可以用type()看下housing和housing.values的不同
#dataFrame不好直接加特征，需要绕一下
housing_values=np.c_[housing.values, housing_category_onehot.toarray()]
housing_fixed = pd.DataFrame(
housing_values,
columns=list(housing.columns)+['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
index=housing.index)
housing_fixed.head()

housing_fixed.describe()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
housing_complete_values=scaler.fit_transform(housing_fixed.values)
housing_complete = pd.DataFrame(
housing_complete_values,
columns=list(housing.columns)+['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
index=housing.index)
housing_complete.describe()


# ##################
test_ratio=0.2 #打算把数据中%20的地区放入测试集
shuffled_indices = np.random.permutation(len(housing_complete)) # 打乱一下顺序
test_set_size = int(len(housing_complete) * test_ratio)
test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]# 获得对应的下标
#一般来说，把完全处理好的特征集记作X，目标集记作y
X_train=housing_complete.iloc[train_indices]
X_test=housing_complete.iloc[test_indices]
y_train=housing_target.iloc[train_indices]
y_test=housing_target.iloc[test_indices]
# 看看是不是一一对应

len(housing_complete)

# 输出：20640

len(X_train)
# # 输出：16512
#
len(X_test)
# # 输出：4128

# 4128+16512=20640

# 选择，训练并评估模型
# 加州房价篇是入门篇，旨在让大家对机器学习的整个过程有一个全面的认识，广度和深度之间往往无法兼顾，模型的原理我在后面的篇章中会有详细的阐述

# 机器学习经过这么多年的发展，前人已经为我们准备好了许多久经磨砺，却又经久不衰的模型，比如随机森林(RandomForest)模型，我们拿来就能用

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)#用训练集训练模型
# 利用测试集的前十个地区，我们来看看效果

forest_reg.predict(X_test[:10])
# 输出：图片

y_test[:10].values
# 输出：图片

# 嗯，除了第一个地区差距有点大之外，还不赖，不过我们就只能用肉眼来判断一个模型的好坏吗？

# 当然不是，现在一般用均方根误差(RMSE)来表示一个模型的好坏,感兴趣的朋友可以搜索均方根误差，看看具体的公式，简而言之，得出的数字越小，模型越好，数字越大，模型越差，我们来试试

#   线性回归 是最简单的

from sklearn.metrics import mean_squared_error
test_predict=forest_reg.predict(X_test)
MSE=mean_squared_error(test_predict,y_test)
RMSE=np.sqrt(MSE)
RMSE
# 输出： 50729

y_test.describe()
# 输出：图片

# 测试集的数据中，房价最大值有50万，最小值2万，而模型的误差在5万，这个结果只能说是一般
#
# 但是仔细想想，我们才读了三篇文章，就已经几乎能做到一整个拿着高薪的分析师团队耗费大量时间才能做到的事情，这正是机器学习的魅力所在
#
# 加州房价篇到此结束，下一个篇章：手写数字识别篇
#
# 对应源码(需下载后查看)
#
# 对应视频
#
# 转载请注明出处
#
# 标签
# 回归模型 入门级