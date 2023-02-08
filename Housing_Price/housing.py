import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import missingno
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


df = pd.read_csv('housing.csv')
# how to plot the correlation matrix of the median housing price column in the df and the other columns
# print(df.corr()['median_house_value'].sort_values(ascending=False))

# another way to plot the correlation matrix using matplotlib
# corr_matrix = df.corr()
# plt.matshow(corr_matrix)
# plt.show()
# show the label as well
# plt.matshow(corr_matrix)
# df.corr() is depcreacated so use df.corr().index instead
# how to disable pandas warning
