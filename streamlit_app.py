import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import streamlit as st

st.header("Breast Cancer Detection.")

st.write("Datasets [ScikitLearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)")

# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# loading the data to a dataframe 
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# adding the target column to the data frame
data_frame['label'] = breast_cancer_dataset.target

st.dataframe(data_frame.describe())

X = data_frame.drop(columns = 'label', axis = 1)
Y = data_frame['label']

st.bar_chart(sns.countplot(data_frame['label'], label = 'count'))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size = 0.2, random_state = 0)

model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = RandomForestClassifier()

model1.fit(X_train,Y_train)
model2.fit(X_train,Y_train)
model3.fit(X_train,Y_train)

X_train_prediction1 = model1.predict(X_train)
training_data_accuracy1 = accuracy_score(Y_train, X_train_prediction1)

X_train_prediction2 = model2.predict(X_train)
training_data_accuracy2 = accuracy_score(Y_train, X_train_prediction2)

X_train_prediction3 = model3.predict(X_train)
training_data_accuracy3 = accuracy_score(Y_train, X_train_prediction3)

st.subheader('Accuracy of each Algorithm')
st.write('Accuracy on training data using LogisticRegression = ', training_data_accuracy1,"%")
st.write('Accuracy on training data using DecisionMakingClassifier = ', training_data_accuracy2,"%")
st.write('Accuracy on training data using RandomForerstClassifier = ', training_data_accuracy3,"%")

# accuracy on test data
X_test_prediction1 = model1.predict(X_test)
test_data_accuracy1 = accuracy_score(Y_test, X_test_prediction1)
test_data_precision1  = precision_score(Y_test, X_test_prediction1)
test_data_recall_score1 = recall_score(Y_test, X_test_prediction1)

X_test_prediction2 = model2.predict(X_test)
test_data_accuracy2 = accuracy_score(Y_test, X_test_prediction2)
test_data_precision2  = precision_score(Y_test, X_test_prediction2)
test_data_recall_score2 = recall_score(Y_test, X_test_prediction2)

X_test_prediction3 = model1.predict(X_test)
test_data_accuracy3 = accuracy_score(Y_test, X_test_prediction3)
test_data_precision3  = precision_score(Y_test, X_test_prediction3)
test_data_recall_score3 = recall_score(Y_test, X_test_prediction3)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, X_test_prediction1)
st.subheader("Confusion Matrix")
st.write(cm)

st.subheader("Accuracy, Precision, and Recall of LogisticRegression")
st.write('Accuracy on the test data using LogisticRegression = ', test_data_accuracy1,'%')
st.write('Precision on the test data using LogisticRegression = ', test_data_precision1,'%')
st.write('Recall Score on the test data using LogisticRegression = ', test_data_recall_score1,'%')

st.subheader("Accuracy, Precision, and Recall of DecisionForestClassifier")
st.write('Accuracy on the test data using DecisionForestClassifier = ', test_data_accuracy2,'%')
st.write('Precision on the test data using DecisionForestClassifier = ', test_data_precision2,'%')
st.write('Recall Score on the test data using DecisionForestClassifier = ', test_data_recall_score2,'%')

st.subheader("Accuracy, Precision, and Recall of RandomForestClassifier")
st.write('Accuracy on the test data using RandomForestClassifier = ', test_data_accuracy3,'%')
st.write('Precision on the test data using LogisticRegression = ', test_data_precision1,'%')
st.write('Recall Score on the test data using LogisticRegression = ', test_data_recall_score1,'%')

import streamlit as st
st.subheader('Prediction')
input_data = st.text_input('Enter the features:',"10.95,21.35,71.9,371.1,0.1227,0.1218,0.1044,0.05669,0.1895,0.0687,0.2366,1.428,1.822,16.97,0.008064,0.01764,0.02595,0.01037,0.01357,0.00304,12.84,35.34,87.22,514,0.1909,0.2698,0.4023,0.1424,0.2964,0.09606")
#st.text_input("Input", "အမုန်းမပွားရဘူးနော်")
#st.write(type(input_data))
input_data = input_data.strip().split(",")
input_data = np.asarray([float(i) for i in input_data]).reshape(1, -1)


#input_data_as_numpy_array = np.asarray(input_data)
#st.write(input_data_as_numpy_array)
# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data.reshape(1,-1)

prediction = model1.predict(input_data_reshaped)

if (prediction[0] == 0):
   #print(f'This Cancer is Malignant, testing by LogistiRegression.')
   st.write('This Cancer is Malignant, testing by LogistiRegression.')
else:
   #print(f'This Cancer is Benign, testing by LogisticRegression.')
   st.write('This Cancer is Malignant, testing by LogistiRegression.')
   
   
prediction = model2.predict(input_data_reshaped)
print(f' Using DecisionTreeClassifier : ',prediction)

if (prediction == 0):
  st.write(f' This Cancer is Malignant, testing by DecisionTreeClassifier.')
else:
  st.write(f' This Cancer is Benign, testing by DecisionTreeClassifier.')


prediction = model3.predict(input_data_reshaped)
print(f' Using DecisionTreeClassifier : ',prediction)

if (prediction == 0):
  st.write(f' This Cancer is Malignant, testing by RandomForestClassifier.')
else:
  st.write(f' This Cancer is Benign, testing by RandomForestClassifier.')
