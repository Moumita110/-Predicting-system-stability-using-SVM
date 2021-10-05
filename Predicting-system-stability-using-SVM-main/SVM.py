# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#printing confusion matrix
print("Here is the confusion matrix")
print(cm)

print("\nHere is the accuracy result of this model")
accu=accuracy_score(y_test, y_pred)
print(accu)


#user given input
input_string = input("Enter Temparature, Moisture in Oil, Moisture in paper, Oil level in sequence: ")
userList = input_string.split()
print("user list is ", userList)


#a=int(classifier.predict(sc.transform([90,12,0.5,80])))

a=int(classifier.predict(sc.transform(userList)))

#prediction on the basis of user given percentage 
if a==0:
    print("You system is perfect")
elif a==1:
    print("Oil leakage")
elif a==2:
    print("Temparature")
elif a==3:
    print("Moisture in oil")
else:
    print("Moisture in paper")


