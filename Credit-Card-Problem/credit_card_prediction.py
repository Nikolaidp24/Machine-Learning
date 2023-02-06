import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# read in the data from local directory
application = pd.read_csv('application_record.csv')
credit = pd.read_csv('credit_record.csv')

# using dict replace to map values to data frame
overdue = {'2': 1, '3': 1, '4': 1, '5': 1, 'X': 0, 'C': 0, 'Q': 0, '0': 0, '1': 0}
credit.replace({'STATUS': overdue}, inplace=True)

# categorize the status column
credit_new = credit.groupby(by='ID', as_index=False)['STATUS'].sum()
credit_new.loc[credit_new.STATUS > 0, 'STATUS'] = 'NG'
credit_new.loc[credit_new.STATUS == 0, 'STATUS'] = 'Good'

# merge datasets
df = pd.merge(application, credit_new, how='left', on='ID')
df = df.dropna()

# pre-categorizing certain columns
df.loc[(df.NAME_INCOME_TYPE == 'Pensioner') | (df.NAME_INCOME_TYPE == 'Student'), 'NAME_INCOME_TYPE'] = 'State servant'
df.loc[(df.NAME_EDUCATION_TYPE == 'Incomplete higher') | (df.NAME_EDUCATION_TYPE == 'Academic degree') | (df.NAME_EDUCATION_TYPE == 'Lower secondary'), 'NAME_EDUCATION_TYPE'] = 'Mid to Low'
df.loc[(df.NAME_HOUSING_TYPE == 'Office apartment') | (df.NAME_HOUSING_TYPE == 'Co-op apartment'), 'NAME_HOUSING_TYPE'] = 'Rented apartment'
df.loc[(df.OCCUPATION_TYPE == 'Laborers') | (df.OCCUPATION_TYPE == 'Sales staff') | (df.OCCUPATION_TYPE == 'Drivers') | (df.OCCUPATION_TYPE == 'Cooking staff') | (df.OCCUPATION_TYPE == 'Security staff') | (df.OCCUPATION_TYPE == 'Cleaning staff') | (df.OCCUPATION_TYPE == 'Low-skill Laborers') | (df.OCCUPATION_TYPE == 'Waiters/barmen staff'), 'OCCUPATION_TYPE'] = 'Working class'
df.loc[(df.OCCUPATION_TYPE == 'Managers') | (df.OCCUPATION_TYPE == 'Accountants') | (df.OCCUPATION_TYPE == 'Medicine staff') | (df.OCCUPATION_TYPE == 'Secretaries') | (df.OCCUPATION_TYPE == 'HR staff') | (df.OCCUPATION_TYPE == 'Realty agents'), 'OCCUPATION_TYPE'] = 'Office class'
df.loc[(df.OCCUPATION_TYPE == 'Core staff') | (df.OCCUPATION_TYPE == 'High skill tech staff') | (df.OCCUPATION_TYPE == 'Private service staff') | (df.OCCUPATION_TYPE == 'IT staff'), 'OCCUPATION_TYPE'] = 'IT'

# transform data based on categories
col_trans = make_column_transformer((OneHotEncoder(), ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE']), remainder='passthrough')

# defining x and y for training
x = df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED', 'STATUS'], axis='columns')
y = df.STATUS.astype('category')

# OneHotEncoder to transform data and use SMOTE to balance the classifier
x_r = col_trans.fit_transform(x)
x_balanced, y_balanced = SMOTE().fit_resample(x_r,y)
x_balanced = pd.DataFrame(x_balanced)

# use train_test_split to evaluate model
x_train, x_test, y_train, y_test = train_test_split(x_balanced, y_balanced, random_state=0)

# logistic regression
logreg = LogisticRegression(solver='lbfgs')
print(f'Logistic Regression Cross Validation Score: {cross_val_score(logreg, x_balanced, y_balanced, cv=5, scoring="accuracy").mean()*100}%')
logreg.fit(x_train, y_train)
y_pred_logreg = logreg.predict(x_test)
print(f'Logistic Regression Accuracy Score using train_test_split: {metrics.accuracy_score(y_pred_logreg, y_test)*100}%')
print('Confusion Matrix:')
print(metrics.confusion_matrix(y_test, y_pred_logreg))

# K nearest neighbors
knn = KNeighborsClassifier(n_neighbors=15)
print(f'K Nearest Neighbors Cross Validation Score: {cross_val_score(knn, x_balanced, y_balanced, cv=5, scoring="accuracy").mean()*100}%')
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
print(f'K Nearest Neighbors Accuracy Score using train_test_split: {metrics.accuracy_score(y_pred_knn, y_test)*100}%')
print('Confusion Matrix:')
print(metrics.confusion_matrix(y_test, y_pred_knn))

# decision tree classifier
dtc = DecisionTreeClassifier(max_depth=10)
print(f'Decision Tree Classifier Cross Validation Score: {cross_val_score(dtc, x_balanced, y_balanced, cv=5, scoring="accuracy").mean()*100}%')
dtc.fit(x_train, y_train)
y_pred_dtc = dtc.predict(x_test)
print(f'Decision Tree Classifier Accuracy Score using train_test_split: {metrics.accuracy_score(y_pred_dtc, y_test)*100}%')
print('Confusion Matrix:')
print(metrics.confusion_matrix(y_test, y_pred_dtc))


