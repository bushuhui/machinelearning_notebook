"""
发散型条形图 （Diverging Bars）
如果您想根据单个指标查看项目的变化情况，并可视化此差异的顺序和数量，
那么散型条形图 （Diverging Bars） 是一个很好的工具。 它有助于快速区分数据中组的性能，
并且非常直观，并且可以立即传达这一点。
"""
from matplotlib import patches
from scipy.spatial import ConvexHull
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

# Prepare Data
df = pd.read_csv('/Users/zuokuijun/PycharmProjects/PythonPlot/data/mtcars.csv')
x = df.loc[:, ['mpg']]
df['mpg_z'] = (x - x.mean())/x.std()
df['colors'] = ['red' if x < 0 else 'green' for x in df['mpg_z']]
df.sort_values('mpg_z', inplace=True)
df.reset_index(inplace=True)

# Draw plot
plt.figure(figsize=(14,10), dpi= 80)
plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z, color=df.colors, alpha=0.4, linewidth=5)

# Decorations
plt.gca().set(ylabel='$Model$', xlabel='$Mileage$')
plt.yticks(df.index, df.cars, fontsize=12)
plt.title('Diverging Bars of Car Mileage', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()

