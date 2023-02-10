import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import missingno
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin


# getting overview of the data
df = pd.read_csv('housing.csv')
df_viz = df.iloc[:, 0:-1]
df_viz.hist(bins=50, figsize=(30,15))
plt.show()

corr_matrix = df_viz.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df_viz.corr(), interpolation=None)
fig.colorbar(cax)
ax.tick_params(axis="x", labelbottom=True, labeltop=False)
ax.set_xticklabels([''] + df_viz.columns.tolist(), rotation=90)
ax.set_yticklabels([''] + df_viz.columns.tolist())
plt.show()

# visualizing the heat map and combine it into the actual map of california
california_img=mpimg.imread('california.png')
df_viz.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
            s=df_viz['population']/100, c=df_viz['median_house_value'], cmap=plt.get_cmap("jet"),
            colorbar=True, sharex=False, label='population')
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.legend()
plt.show()

df_exp = df.copy()
x = df_exp.drop('median_house_value', axis=1)
y = df.pop('median_house_value')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# create an option to use later on to add either of the two columns described below
class Add_extra(BaseEstimator, TransformerMixin):
    def __init__(self):
        return
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X1 = X.copy()
        X1['population_per_household'] = X1['population'] / X1['households']
        X1['beds_per_room'] = X1['total_bedrooms'] / X1['total_rooms']
        return X1


# make column transformer before making pipelines
simp = SimpleImputer(strategy='median')
ohe = OneHotEncoder(handle_unknown='ignore')
col_trans = make_column_transformer((simp, ['total_bedrooms', 'beds_per_room']),
                                   (ohe, ['ocean_proximity']), remainder='passthrough')

# make pipeline for linreg
linreg_pre = LinearRegression()
pipe_linreg_pre = make_pipeline(Add_extra(), col_trans, linreg_pre)
pipe_linreg_pre.fit(x_train, y_train)
y_pred_linreg_pre = pipe_linreg_pre.predict(x_test)
print(f'Metrics RMSE: {metrics.mean_squared_error(y_test, y_pred_linreg_pre, squared=False)}')
print(f'Cross Validation RMSE: {np.sqrt(-cross_val_score(pipe_linreg_pre, x, y, cv=5, scoring="neg_mean_squared_error"))}')

# make pipeline for dtr
dtr_pre = DecisionTreeRegressor()
pipe_dtr_pre = make_pipeline(Add_extra(), col_trans, dtr_pre)
pipe_dtr_pre.fit(x_train, y_train)
y_pred_dtr_pre = pipe_dtr_pre.predict(x_test)
print(f'Metrics RMSE: {metrics.mean_squared_error(y_test, y_pred_dtr_pre, squared=False)}')
print(f'Cross Validation RMSE: {np.sqrt(-cross_val_score(pipe_dtr_pre, x, y, cv=5, scoring="neg_mean_squared_error"))}')

# make pipeline for SVR
svr_pre = SVR()
pipe_svr_pre = make_pipeline(Add_extra(), col_trans, svr_pre)
pipe_svr_pre.fit(x_train, y_train)
y_pred_svr_pre = pipe_svr_pre.predict(x_test)
print(f'Metrics RMSE: {metrics.mean_squared_error(y_test, y_pred_svr_pre, squared=False)}')
print(f'Cross Validation RMSE: {np.sqrt(-cross_val_score(pipe_svr_pre, x, y, cv=5, scoring="neg_mean_squared_error", n_jobs=-1))}')

# make pipeline for rfr
rfr_pre = RandomForestRegressor()
pipe_rfr_pre = make_pipeline(Add_extra(), col_trans, rfr_pre)
pipe_rfr_pre.fit(x_train, y_train)
y_pred_rfr_pre = pipe_rfr_pre.predict(x_test)
print(f'Metrics RMSE: {metrics.mean_squared_error(y_test, y_pred_rfr_pre, squared=False)}')
print(f'Cross Validation RMSE: {np.sqrt(-cross_val_score(pipe_rfr_pre, x, y, cv=5, scoring="neg_mean_squared_error", n_jobs=-1))}')

params_linreg = {}
params_linreg['linearregression__fit_intercept'] = [True, False]
grid_linreg = GridSearchCV(pipe_linreg_pre, params_linreg, cv=5, scoring='neg_mean_squared_error')
grid_linreg.fit(x, y)

params_dtr = {}
params_dtr['decisiontreeregressor__max_depth'] = [6, 7, 8, 9, 10, 11, 12, 13, 14]
params_dtr['decisiontreeregressor__min_samples_leaf'] = [16, 17, 18, 19, 20, 21, 22, 23, 24]
grid_dtr = GridSearchCV(pipe_dtr_pre, params_dtr, cv=5, scoring='neg_mean_squared_error')
grid_dtr.fit(x, y)

params_svr = [{'svr__kernel': ['linear'], 'svr__C': [0.1, 1, 10, 100]}, {'svr__kernel': ['rbf'], 'svr__C': [1, 10, 100], 'svr__gamma': [0.01, 0.1, 0.2]}]
grid_svr = GridSearchCV(pipe_svr_pre, params_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_svr.fit(x, y)

params_rfr = {}
params_rfr['randomforestregressor__n_estimators'] = [50, 100, 150]
params_rfr['randomforestregressor__min_samples_leaf'] = [10, 20, 30]
grid_rfr = GridSearchCV(pipe_rfr_pre, params_rfr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_rfr.fit(x, y)

linreg = LinearRegression()
dtr = DecisionTreeRegressor(max_depth=8, min_samples_leaf=20)
svr = SVR(C=100, kernel='linear')
rfr = RandomForestRegressor(n_estimators=100, min_samples_leaf=20)
pipe_linreg = make_pipeline(Add_extra(), col_trans, linreg)
pipe_dtr = make_pipeline(Add_extra(), col_trans, dtr)
pipe_svr = make_pipeline(Add_extra(), col_trans, svr)
pipe_rfr = make_pipeline(Add_extra(), col_trans, rfr)
model_list = [pipe_linreg, pipe_dtr, pipe_svr, pipe_rfr]
final_rmse = []
for model in model_list:
    rmse = np.sqrt(-cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error').mean())
    final_rmse.append(rmse)
print(final_rmse)

model_list_pre = [pipe_linreg_pre, pipe_dtr_pre, pipe_svr_pre, pipe_rfr_pre]
before_rmse = []
for model_before in model_list_pre:
    rmse_before = np.sqrt(-cross_val_score(model_before, x, y, cv=5, scoring='neg_mean_squared_error').mean())
    before_rmse.append(rmse_before)
print(before_rmse)

rmse_dict = {'LinearRegression': [final_rmse[0], before_rmse[0]], 'DecisionTreeRegressor': [final_rmse[1], before_rmse[1]], 'SVR': [final_rmse[2], before_rmse[2]], 'RandomForestRegressor': [final_rmse[3], before_rmse[3]]}
df_result = pd.DataFrame.from_dict(data=rmse_dict, orient='index')
df_result.plot(kind='barh', grid=True, figsize=(5, 3))
plt.legend(['After Tuning', 'Before Tuning'])
plt.show()