import pandas as pd
import numpy as np
import missingno
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from imblearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')


# custom function for processing data
def cus_trans(df):
    df.loc[df.GenHlth > 3, 'GenHlth'] = 3
    df.loc[df.MentHlth <= 5, 'MentHlth'] = 0
    df.loc[(df.MentHlth > 5) & (df.MentHlth <= 10), 'MentHlth'] = 1
    df.loc[df.MentHlth > 10, 'MentHlth'] = 2
    df.loc[df.PhysHlth <= 5, 'PhysHlth'] = 0
    df.loc[(df.PhysHlth > 5) & (df.PhysHlth <= 10), 'PhysHlth'] = 1
    df.loc[df.PhysHlth > 10, 'PhysHlth'] = 2
    df.loc[df.Education <= 3, 'Education'] = 4
    df.loc[df.Income < 4, 'Income'] = 0
    df.loc[(df.Income >= 4) & (df.Income < 7), 'Income'] = 1
    df.loc[df.Income >= 7, 'Income'] = 2
    df.loc[df.Age < 6, 'Age'] = 0
    df.loc[(df.Age >= 6) & (df.Age < 10), 'Age'] = 1
    df.loc[df.Age >= 10, 'Age'] = 2
    return df


# combine custom function and OrdinalEncoder
y = df.Diabetes_012
x = df.drop(['Diabetes_012'], axis='columns')
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.4)
ode = OrdinalEncoder()
custom_trans = FunctionTransformer(cus_trans)
col_trans = make_column_transformer((custom_trans, ['Age', 'Income', 'Education', 'PhysHlth', 'MentHlth', 'GenHlth']),
                                    (ode, ['Age', 'Income', 'Education', 'PhysHlth', 'MentHlth', 'GenHlth']),
                                    remainder='passthrough')

# making pipeline
knn = KNeighborsClassifier(n_neighbors=5)
pipe_knn = make_pipeline(SMOTE(random_state=14), col_trans, knn)

# evaluate the pipeline for knn
pipe_knn.fit(x_train, y_train)
y_pred_knn = pipe_knn.predict(x_test)
print(f'Cross Validation Accuracy: {cross_val_score(pipe_knn, x, y, cv=5, scoring="accuracy").mean()*100}%')
print(f'Metrics Accuracy: {metrics.accuracy_score(y_pred_knn, y_test)*100}%')
cm = metrics.confusion_matrix(y_test, y_pred_knn, labels=knn.classes_)
display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
display.plot()
plt.show()

# making pipeline for logreg
logreg = LogisticRegression(solver='liblinear', C=0.1)
pipe_logreg = make_pipeline(SMOTE(random_state=14), col_trans, logreg)

# evaluate the pipeline for logreg
pipe_logreg.fit(x_train, y_train)
y_pred_logreg = pipe_logreg.predict(x_test)
print(f'Cross Validation Accuracy Score: {cross_val_score(pipe_logreg, x, y, cv=5, scoring="accuracy", n_jobs=-1).mean()*100}%')
print(f'Metrics Accuracy Score: {metrics.accuracy_score(y_pred_logreg, y_test)*100}%')
cm = metrics.confusion_matrix(y_test, y_pred_logreg, labels=logreg.classes_)
display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
display.plot()
plt.show()

# making pipeline for dtc
dtc = DecisionTreeClassifier(max_depth=15, min_samples_leaf=5)
pipe_dtc = make_pipeline(SMOTE(random_state=14), col_trans, dtc)

# evaluate the pipeline for dtc
pipe_dtc.fit(x_train, y_train)
y_pred_dtc = pipe_dtc.predict(x_test)
print(f'Cross Validation Accuracy Score: {cross_val_score(pipe_dtc, x, y, cv=5, scoring="accuracy").mean()*100}%')
print(f'Metrics Accuracy Score: {metrics.accuracy_score(y_pred_dtc, y_test)*100}%')
cm = metrics.confusion_matrix(y_test, y_pred_dtc, labels=dtc.classes_)
display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dtc.classes_)
display.plot()
plt.show()