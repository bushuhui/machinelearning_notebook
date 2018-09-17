# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext_format_version: '1.2'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.5.2
# ---

# # Exercise - 交通事故理赔审核预测
#
#
# 这个比赛的链接：http://sofasofa.io/competition.php?id=2
#
#
# * 任务类型：二元分类
#
# * 背景介绍：在交通摩擦（事故）发生后，理赔员会前往现场勘察、采集信息，这些信息往往影响着车主是否能够得到保险公司的理赔。训练集数据包括理赔人员在现场对该事故方采集的36条信息，信息已经被编码，以及该事故方最终是否获得理赔。我们的任务是根据这36条信息预测该事故方没有被理赔的概率。
#
# * 数据介绍：训练集中共有200000条样本，预测集中有80000条样本。 
# ![data_description](images/data_description.png)
#
# * 评价方法：Precision-Recall AUC
#

# ## Demo code
#

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# %matplotlib inline

# read data
homePath = "data"
trainPath = os.path.join(homePath, "train.csv")
testPath = os.path.join(homePath, "test.csv")
submitPath = os.path.join(homePath, "sample_submit.csv")
trainData = pd.read_csv(trainPath)
testData = pd.read_csv(testPath)
submitData = pd.read_csv(submitPath)

# 参照数据说明，CaseID这列是没有意义的编号，因此这里将他丢弃。
#
# ~drop()函数：axis指沿着哪个轴，0为行，1为列；inplace指是否在原数据上直接操作
#

# 去掉没有意义的一列
trainData.drop("CaseId", axis=1, inplace=True)
testData.drop("CaseId", axis=1, inplace=True)

# # 快速了解数据
#
# ~head()：默认显示前5行数据，可指定显示多行，例如.head(15)显示前15行
#

trainData.head(15)


# 显示数据简略信息，可以每列有多少非空的值，以及每列数据对应的数据类型。
#
#

trainData.info()


# ~hist():绘制直方图，参数figsize可指定输出图片的尺寸。
#

trainData.hist(figsize=(20, 20))


# 想要了解特征之间的相关性，可计算相关系数矩阵。然后可对某个特征来排序。
#
#

corr_matrix = trainData.corr()
corr_matrix["Evaluation"].sort_values(ascending=False) # ascending=False 降序排列

# 从训练集中分离标签

y = trainData['Evaluation']
trainData.drop("Evaluation", axis=1, inplace=True)

# 使用K-Means训练模型
#
# KMeans()：
# * `n_clusters`指要预测的有几个类；
# * `init`指初始化中心的方法，默认使用的是`k-means++`方法，而非经典的K-means方法的随机采样初始化，当然你可以设置为random使用随机初始化；
# * `n_jobs`指定使用CPU核心数，-1为使用全部CPU。

# +
# do k-means
from sklearn.cluster import KMeans
est = KMeans(n_clusters=2, init="k-means++", n_jobs=-1)
est.fit(trainData, y)

y_train = est.predict(trainData)
y_pred = est.predict(testData)

# 保存预测的结果
submitData['Evaluation'] = y_pred
submitData.to_csv("submit_data.csv", index=False)

# +
# calculate accuracy
from sklearn.metrics import accuracy_score

acc_train = accuracy_score(y, y_train)
print("acc_train = %f" % (acc_train))
# -

# ## 随机森林
#
# 使用K-means可能得到的结果没那么理想。在官网上，举办方给出了两个标杆模型，效果最好的是随机森林。以下是代码，读者可以自己测试。
#
#

# +
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 读取数据
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
submit = pd.read_csv("data/sample_submit.csv")

# 删除id
train.drop('CaseId', axis=1, inplace=True)
test.drop('CaseId', axis=1, inplace=True)

# 取出训练集的y
y_train = train.pop('Evaluation')

# 建立随机森林模型
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(train, y_train)
y_pred = clf.predict_proba(test)[:, 1]

# 输出预测结果至my_RF_prediction.csv
submit['Evaluation'] = y_pred
submit.to_csv('my_RF_prediction.csv', index=False)



# +
# freature importances
print(clf.feature_importances_)

# Train accuracy
from sklearn.metrics import accuracy_score
y_train_pred = clf.predict(train)
print(y_train_pred)

acc_train = accuracy_score(y_train, y_train_pred)
print("acc_train = %f" % (acc_train))
