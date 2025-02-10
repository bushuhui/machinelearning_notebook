"""
日历热力图 （Calendar Heat Map）
与时间序列相比，日历地图是可视化基于时间的数据的备选和不太优选的选项。 虽然可以在视觉上吸引人，但数值并不十分明显。 然而，它可以很好地描绘极端值和假日效果。
"""
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings(action='once')
import init_parameter as ip
import statsmodels.tsa.stattools as stattools
import numpy as np
import matplotlib as mpl
import calmap
ip.init()

# Import Data
df = pd.read_csv('/Users/zuokuijun/PycharmProjects/PythonPlot/data/yahoo.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# Plot
plt.figure(figsize=(16,10), dpi= 80)
calmap.calendarplot(df['2014']['VIX.Close'], fig_kws={'figsize': (16,10)}, yearlabel_kws={'color':'black', 'fontsize':14}, subplot_kws={'title':'Yahoo Stock Prices'})
plt.show()
