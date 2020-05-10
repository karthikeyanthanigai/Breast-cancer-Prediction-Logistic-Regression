#Import the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the Dataset
Dataset = pd.read_csv("breast_cancer_dataset.csv")
X = Dataset.iloc[:, :-1].values
y = Dataset.iloc[:, [-1]].values

#feature scale our independent variables 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=32, shuffle=True)

#Logistic Regression model. 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=1000, random_state = 32, multi_class='ovr', solver='liblinear')
classifier.fit(X_train, y_train)

#predict our test set
y_pred = classifier.predict(X_test)

#evaluating our model.
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
acc_train = accuracy_score(y_train, classifier.predict(X_train))
f1_train = f1_score(y_train, classifier.predict(X_train), average= 'weighted')

print("Traing set results")
print("ACCURACY for train set",acc_train)
print("F1 SCORE for train set",f1_train)

#evaluate our test set
acc_test = accuracy_score(y_test, y_pred)
f1_test = f1_score(y_test, y_pred, average= 'weighted')

print("Test set results")
print("ACCURACY for test set",acc_test)
print("F1 SCORE for test set",f1_test)

#Confusion Matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)
