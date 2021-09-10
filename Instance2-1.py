# -*- UESTC 014 -*-
"""
作者：RW
日期：2021年09月08日
功能：房价预测
"""
# -*-数据下载、存储、导入和可视化
import sys

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "Instance2-1"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
# print("Save path is:\n",IMAGES_PATH)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()  # 调整参数，填充整个图像区域
    plt.savefig(path, format=fig_extension, dpi=resolution)


import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):  # 判断是否是一个目录
        os.makedirs(housing_path)  # 创建目录
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)  # 复制URL到本地文件
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# fetch_housing_data()

import pandas as pd


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    # print("House information is：\n", pd.read_csv(csv_path).head())
    return pd.read_csv(csv_path)


housing = load_housing_data()
# housing.info()
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())
plt.figure(1)
housing.hist(bins=50, figsize=(20, 15))  # bins:每个直方图的柱数，即所要分的组  figsize(行间距，列间距)
plt.show()
# -*-创建测试集
# np.random.seed(1)


# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))  # np.random.permutation：随机排序
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     # print("final data:\n", data.iloc[train_indices], data.iloc[test_indices])
#     return data.iloc[train_indices], data.iloc[test_indices]

# 纯随机抽样
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# 分层抽样
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
plt.figure(2)
housing["income_cat"].hist()
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # 分层抽样
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_train_set["income_cat"].value_counts() / len(strat_test_set))
print(housing["income_cat"].value_counts() / len(housing))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


