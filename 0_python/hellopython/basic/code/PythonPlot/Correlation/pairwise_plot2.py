
"""
矩阵图 （Pairwise Plot）
矩阵图是探索性分析中的最爱，用于理解所有可能的数值变量对之间的关系。 它是双变量分析的必备工具。
"""
# !pip install brewer2mpl
import numpy as np
import pandas as pd
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

import init_parameter as ip

ip.init()
# Load Dataset
df = pd.read_csv('/Users/zuokuijun/PycharmProjects/PythonPlot/data/iris.csv')

plt.figure(figsize=(10,8), dpi= 80)
sns.pairplot(df, kind="reg", hue="species")
plt.show()