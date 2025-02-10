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
g = sns.catplot("alive", col="deck", col_wrap=4,
                data=titanic[titanic.deck.notnull()],
                kind="count", height=3.5, aspect=.8,
                palette='tab20')

##fig.suptitle('sf')
plt.show()