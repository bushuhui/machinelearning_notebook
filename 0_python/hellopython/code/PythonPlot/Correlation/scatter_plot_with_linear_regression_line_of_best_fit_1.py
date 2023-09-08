"""
带线性回归最佳拟合线的散点图 （Scatter plot with linear regression line of best fit）
如果你想了解两个变量如何相互改变，那么最佳拟合线就是常用的方法。
下图显示了数据中各组之间最佳拟合线的差异。 要禁用分组并仅为整个数据集绘制一条最佳拟合线，
请从下面的sns.lmplot（）调用中删除hue ='cyl'参数。
@Author zuokuijun
@Date   2021-08-11
"""
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
df = pd.read_csv('/Users/zuokuijun/PycharmProjects/PythonPlot/data/mpg_ggplot2.csv')
df_select = df.loc[df.cyl.isin([4,8]), :]

# Plot
sns.set_style("white")
gridobj = sns.lmplot(x="displ", y="hwy", hue="cyl", data=df_select,
                     height=7, aspect=1.6, robust=True, palette='tab10',
                     scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))

# Decorations
gridobj.set(xlim=(0.5, 7.5), ylim=(0, 50))
plt.title("Scatterplot with line of best fit grouped by number of cylinders", fontsize=20)
plt.show()


