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

# # 混淆矩阵(confusion matrix)
#
# 混淆矩阵是用来总结一个分类器结果的矩阵。对于k元分类，其实它就是一个$k \times k$的表格，用来记录分类器的预测结果。
#
# 对于最常见的二元分类来说，它的混淆矩阵是2乘2的，如下
# ![confusion_matrix1](images/confusion_matrix1.png)
#
# * `TP` = True Postive = 真阳性
# * `FP` = False Positive = 假阳性
# * `FN` = False Negative = 假阴性
# * `TN` = True Negative = 真阴性
#
# 你要的例子来了。。。比如我们一个模型对15个样本进行预测，然后结果如下。
#
# * 预测值：1    1    1    1    1    0    0    0    0    0    1    1    1    0    1
# * 真实值：0    1    1    0    1    1    0    0    1    0    1    0    1    0    0
#
# ![confusion_matrix2](images/confusion_matrix2.png)
#
#
# 这个就是混淆矩阵。混淆矩阵中的这四个数值，经常被用来定义其他一些度量。
#
#
# ### 准确度
# ```
# Accuracy = (TP+TN) / (TP+TN+FN+TN)
# ```
#
# 在上面的例子中，准确度 = (5+4) / 15 = 0.6
#
#
#
# ### 精度(precision, 或者PPV, positive predictive value) 
# ```
# precision = TP / (TP + FP)
# ```
# 在上面的例子中，精度 = 5 / (5+4) = 0.556
#
#
#
# ### 召回(recall, 或者敏感度，sensitivity，真阳性率，TPR，True Positive Rate) 
#
# ```
# recall = TP / (TP + FN)
# ```
#
# 在上面的例子中，召回 = 5 / (5+2) = 0.714
#
#
#
# ### 特异度(specificity，或者真阴性率，TNR，True Negative Rate)
# ```
# specificity = TN / (TN + FP)
# ```
#
# 在上面的例子中，特异度 = 4 / (4+2) = 0.667
#
#
#
# ### F1-值(F1-score) 
# ```
# F1 = 2*TP / (2*TP+FP+FN) 
# ```
# 在上面的例子中，F1-值 = 2*5 / (2*5+4+2) = 0.625
#
#
#
