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

# ## 数值计算
#
#
# ### （1）对于一个存在在数组，如何添加一个用0填充的边界?
# 例如对一个二维矩阵
# ```
# 10, 34, 54, 23
# 31, 87, 53, 68
# 98, 49, 25, 11
# 84, 32, 67, 88
# ```
#
# 变换成
# ```
#  0,  0,  0,  0,  0, 0
#  0, 10, 34, 54, 23, 0
#  0, 31, 87, 53, 68, 0
#  0, 98, 49, 25, 11, 0
#  0, 84, 32, 67, 88, 0
#  0,  0,  0,  0,  0, 0
# ```
#
# ### （2） 创建一个 5x5的矩阵，并设置值1,2,3,4落在其对角线下方位置
#
#
# ### （3） 创建一个8x8 的矩阵，并且设置成国际象棋棋盘样式（黑可以用0, 白可以用1）
#
#
# ### （4）求解线性方程组
#
# 给定一个方程组，如何求出其的方程解。有多种方法，分析各种方法的优缺点（最简单的方式是消元方）。
#
# 例如
# ```
# 3x + 4y + 2z = 10
# 5x + 3y + 4z = 14
# 8x + 2y + 7z = 20
# ```
#
# 编程写出求解的程序
#
#
# ### （5） 翻转一个数组（第一个元素变成最后一个）
#
#
# ### （6） 产生一个十乘十随机数组，并且找出最大和最小值
#
#
# ## Reference
# * [100 numpy exercises](https://github.com/rougier/numpy-100)
