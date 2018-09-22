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

# # matplotlib
#
#

# ## 1. pyplot
# matplotlib.pyplot is a collection of command style functions that make matplotlib work like MATLAB. Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels, etc. In matplotlib.pyplot various states are preserved across function calls, so that it keeps track of things like the current figure and plotting area, and the plotting functions are directed to the current axes (please note that “axes” here and in most places in the documentation refers to the axes part of a figure and not the strict mathematical term for more than one axis).

# +
# This line configures matplotlib to show figures embedded in the notebook, 
# instead of opening a new window for each figure. More about that later. 
# If you are using an old version of IPython, try using '%pylab inline' instead.
# %matplotlib inline

import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
# -

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])


# For every x, y pair of arguments, there is an optional third argument which is the format string that indicates the color and line type of the plot. The letters and symbols of the format string are from MATLAB, and you concatenate a color string with a line style string. The default format string is ‘b-‘, which is a solid blue line. For example, to plot the above with red circles, you would issue

import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()

# +
import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
# -

# ### [Controlling line properties](https://matplotlib.org/users/pyplot_tutorial.html#controlling-line-properties)
#
# Lines have many attributes that you can set: linewidth, dash style, antialiased, etc; see matplotlib.lines.Line2D. There are several ways to set line properties
#

# ### Working with multiple figures and axes
#
# MATLAB, and pyplot, have the concept of the current figure and the current axes. All plotting commands apply to the current axes. The function gca() returns the current axes (a matplotlib.axes.Axes instance), and gcf() returns the current figure (matplotlib.figure.Figure instance). Normally, you don’t have to worry about this, because it is all taken care of behind the scenes. Below is a script to create two subplots.
#
#

# +
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()
# -

# ## 2. Image 

# +
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# load image
img=mpimg.imread('example.png')

imgplot = plt.imshow(img)

# -

# ### Applying pseudocolor schemes to image plots

lum_img = img[:,:,0]
plt.imshow(lum_img)

# use 'hot' color map
plt.imshow(lum_img, cmap="hot")
plt.colorbar()

# ### Examining a specific data range
#

plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')

# ## References
#
# * [Pyplot tutorial](https://matplotlib.org/users/pyplot_tutorial.html)
# * [Image tutorial](https://matplotlib.org/users/image_tutorial.html)
