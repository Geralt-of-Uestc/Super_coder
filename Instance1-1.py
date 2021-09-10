# -*- UESTC 014 -*-
"""
作者：RW
日期：2021年09月08日
功能：人均GDP与生活满意度的关系
"""

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"

import os
datapath = os.path.join("datasets", "lifesat", "")  # 路径拼接

import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)





def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    # print(oecd_bli.head())
    gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv", thousands=',', delimiter='\t',
                                 encoding='latin1', na_values ="n/a")
    # gdp_per_capita.info()
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    # print(gdp_per_capita.head())
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    # print(full_country_stats.head())
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    # print(full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices].head())
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model

oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')  # 读取数据 thousands:千分位分隔符 1000,00
# oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
# oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
# print(X[1:5])
y = np.c_[country_stats["Life satisfaction"]]
# print(y[1:5])
# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')


# Select a linear model

model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)
plt.plot(X, model.predict(X), color='red', linewidth=3)
print('intercept_:%.3f' % model.intercept_)
print("回归系数为：\n", model.coef_)

# Make a prediction for Cyprus
X_new = [[10582]]  # Cyprus' GDP per capita
X_predict = model.predict(X_new)
plt.scatter(X_new, X_predict, c='g', s=150, marker='o')
print("中国的幸福程度为：\n", model.predict(X_new))  # outputs [[ 5.96242338]]
plt.show()


# import urllib.request
# DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
# os.makedirs(datapath, exist_ok=True)
# for filename in ("oecd_bli_2015.csv", "gdp_per_capita.csv"):
#     print("Downloading", filename)
#     url = DOWNLOAD_ROOT + "datasets/lifesat/" + filename
#     urllib.request.urlretrieve(url, datapath + filename)