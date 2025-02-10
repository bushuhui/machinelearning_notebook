"""
由 seaborn库 提供的分类图可用于可视化彼此相关的2个或更多分类变量的计数分布。
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

# Load Dataset
titanic = pd.read_csv('/Users/zuokuijun/PycharmProjects/PythonPlot/data/titanic.csv')

# Plot
# Plot
sns.catplot(x="age", y="embark_town",
            hue="sex", col="class",
            data=titanic[titanic.embark_town.notnull()],
            orient="h", height=5, aspect=1, palette="tab10",
            kind="violin", dodge=True, cut=0, bw=.2)

# fig.suptitle('sf')
plt.show()