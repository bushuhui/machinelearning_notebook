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

# # Titanic
#
# ## Competition Description
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
#
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
#
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
#
# ## Practice Skills
# * Binary classification
# * Python & SKLearn
#
# ## Data
# The data has been split into two groups:
#
# * training set (train.csv)
# * test set (test.csv)
#
# The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the `ground truth`) for each passenger. Your model will be based on `features` like passengers' gender and class. You can also use feature engineering to create new features.
#
# The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
#
# We also include `gender_submission.csv`, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.
#
# ### Data description
# ![data description1](images/data_description1.png)
# ![data description2](images/data_description2.png)
#
#
# ### Variable Notes
# pclass: A proxy for socio-economic status (SES)
# * 1st = Upper
# * 2nd = Middle
# * 3rd = Lower
#
#
# ## Links
# * [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
