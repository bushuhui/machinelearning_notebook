# !pip install brewer2mpl
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

def init():
    large = 22; med = 16; small = 12
    params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    print(mpl.__version__)  # > 3.0.0
    print(sns.__version__)  # > 0.9.0


