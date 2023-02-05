import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np


df = pd.read_csv('Titanic.csv', sep=',')

# take only essential information and drop the empty age value rows
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
df = df.dropna(how='any')

# mapping sex column to be integer
d = {'male': 1, 'female': 0}
df['Sex'] = df.Sex.apply(lambda x: d[x])

# selecting feature columns and response column
feature_col = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
x = df[feature_col]
y = df.Survived

# use test_train_split of scikit-learn to automatically split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

# getting the best knn neighbor number
score = []
for i in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    score.append(metrics.accuracy_score(y_pred, y_test))
best_knn = np.argmax(score) + 1
KNN_score = np.max(score)
algo_compare = []
algo_compare.append(KNN_score)

# Logistic Regression Algorithm
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred2 = logreg.predict(x_test)
LR = metrics.accuracy_score(y_pred2, y_test)
algo_compare.append(LR)

# GaussianNB Algorithm
gauss = GaussianNB()
gauss.fit(x_train, y_train)
y_pred3 = gauss.predict(x_test)
GR = metrics.accuracy_score(y_pred3, y_test)
algo_compare.append(GR)

# result
compare_result = {'KNeighborsClassifier': algo_compare[0], 'LogisticRegression': algo_compare[1], 'GaussianNB': algo_compare[2]}
print(compare_result)


