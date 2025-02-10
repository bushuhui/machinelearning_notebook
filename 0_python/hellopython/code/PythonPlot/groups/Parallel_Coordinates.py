"""
平行坐标 （Parallel Coordinates）
平行坐标有助于可视化特征是否有助于有效地隔离组。
如果实现隔离，则该特征可能在预测该组时非常有用。
"""
from matplotlib import patches
from scipy.spatial import ConvexHull
from pandas.plotting import parallel_coordinates
import scipy.cluster.hierarchy as shc
import warnings; warnings.simplefilter('ignore')
# !pip install brewer2mpl
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import init_parameter as ip

ip.init()

# Import Data
df_final = pd.read_csv('/Users/zuokuijun/PycharmProjects/PythonPlot/data/diamonds_filter.csv')

# Plot
plt.figure(figsize=(12,9), dpi= 80)
parallel_coordinates(df_final, 'cut', colormap='Dark2')

# Lighten borders
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)

plt.title('Parallel Coordinated of Diamonds', fontsize=22)
plt.grid(alpha=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()