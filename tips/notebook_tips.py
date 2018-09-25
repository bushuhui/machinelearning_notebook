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

# ## Show LaTeX equation
#
#

from IPython.core.display import HTML
HTML("""
<style>

div.cell { /* Tunes the space between cells */
margin-top:1em;
margin-bottom:1em;
}

div.text_cell_render h1 { /* Main titles bigger, centered */
font-size: 2.2em;
line-height:1.4em;
text-align:center;
}

div.text_cell_render h2 { /*  Parts names nearer from text */
margin-bottom: -0.4em;
}


div.text_cell_render { /* Customize text cells */
font-family: 'Times New Roman';
font-size:1.5em;
line-height:1.4em;
padding-left:3em;
padding-right:3em;
}
</style>
""")

from IPython.display import Latex
Latex(r"""\begin{eqnarray}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\
\nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\
\nabla \cdot \vec{\mathbf{B}} & = 0 
\end{eqnarray}""")

# %%latex
\begin{align}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\
\nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\
\nabla \cdot \vec{\mathbf{B}} & = 0
\end{align}

# \begin{align}
# \nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\
# \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\
# \nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\
# \nabla \cdot \vec{\mathbf{B}} & = 0
# \end{align}
#
# \begin{equation}
# E = F \cdot s 
# \end{equation}
#
# \begin{eqnarray}
# F & = & sin(x) \\
# G & = & cos(x)
# \end{eqnarray}
#
# \begin{align}
#     g &= \int_a^b f(x)dx \label{eq1} \\
#     a &= b + c \label{eq2}
# \end{align}
#
# See (\ref{eq1})

# ## Audio
#

from IPython.display import Audio
Audio(url="http://www.nch.com.au/acm/8k16bitpcm.wav")

# +
import numpy as np
max_time = 3
f1 = 220.0
f2 = 224.0
rate = 8000.0
L = 3
times = np.linspace(0,L,rate*L)
signal = np.sin(2*np.pi*f1*times) + np.sin(2*np.pi*f2*times)

Audio(data=signal, rate=rate)
# -

# ## External sites

# + {"scrolled": true}
from IPython.display import IFrame
IFrame('https://jupyter.org', width='100%', height=350)
# -

# ## JupyterLab

# +
import numpy as np
from pprint import pprint

pp = pprint
a = np.array([1, 2, 3])
pp(a)

# -

# ### [jupyter-matplotlib](https://github.com/matplotlib/jupyter-matplotlib)
#
#
# ```
# # Installing Node.js 5.x on Ubuntu / Debian
# curl -sL https://deb.nodesource.com/setup_5.x | sudo -E bash -
# sudo apt-get install -y nodejs
#
# pip install ipympl
#
# # If using JupyterLab
# # Install nodejs: https://nodejs.org/en/download/
# jupyter labextension install @jupyter-widgets/jupyterlab-manager
# jupyter labextension install jupyter-matplotlib
# ```

# ## References
#
# * https://nbviewer.jupyter.org/github/ipython/ipython/blob/master/examples/IPython%20Kernel/Index.ipynb
