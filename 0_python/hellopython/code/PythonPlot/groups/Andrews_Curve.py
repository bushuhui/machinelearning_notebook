"""
安德鲁斯曲线 （Andrews Curve）
安德鲁斯曲线有助于可视化是否存在基于给定分组的数字特征的固有分组。
如果要素（数据集中的列）无法区分组（cyl），那么这些线将不会很好地隔离.
"""
from matplotlib import patches
from scipy.spatial import ConvexHull
import scipy.cluster.hierarchy as shc
import warnings; warnings.simplefilter('ignore')
from pandas.plotting import andrews_curves
# !pip install brewer2mpl
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import init_parameter as ip
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import ConvexHull

ip.init()

# Import
df = pd.read_csv('/Users/zuokuijun/PycharmProjects/PythonPlot/data/mtcars.csv')
df.drop(['cars', 'carname'], axis=1, inplace=True)

# Plot
plt.figure(figsize=(12,9), dpi= 80)
andrews_curves(df, 'cyl', colormap='Set1')

# Lighten borders
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)

plt.title('Andrews Curves of mtcars', fontsize=22)
plt.xlim(-3,3)
plt.grid(alpha=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()