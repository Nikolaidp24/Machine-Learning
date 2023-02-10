# Machine-Learning
Machine learning examples using scikit-learn in python.

## Example One: Titanic
In this example, I will use the data of the famous titanic dataset, using scikit-learn to train a model that will simulate if a person's survival rate on the titanic with given input of the following: what is the given person/people's Sex, which classes did they board on the titanic, their age, their Fare, their sibling/spouse count,  their parent/child count and so on.

This is the first MachineLearning prject of mine, so there might be some flaw along the way.

This is a binary classfication problem.

## Example Two: Credit Card Approval
In this example, I am using two .csv files to read in the credit appication and the applicant's credit score of the past, based on these two files, I will be examing whether an applicant is worthy of the credit card approval or not. Here, a risky behaviour will be marked as bad credit, and the person will not be approved.

This is a binary classification problem.

## Example Three: Diabetes
In this example, I am using the diabetes record of many patients, and their health/personal record. Based on all the feature columns, I will be predicting whether a patient is pre-diabetic, diabetic, or non-diabetic. Please note that there is a severe class imbalance among the three.

One more thing to notice about this example is that I started using more data preprocessing technique with sklearn's built-in modules instead of preprocessing the data using pandas. So the code might look a bit different at first glance.

This is a multi-class classification problem.

## Example Four: Housing Price Prediction
In the example, I have increased the use of plotting in the jupyter notebook to visualize data better, this example utilizes four primary models for evaluation purposes: ***LinearRegression, DecisionTreeRegressor, SVR, and RandomForestRegressor***, I managed to fine tune the model down by quite a bit and the result should be directly visible in the plot.

This is a regression problem.
