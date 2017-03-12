#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy as np
import sklearn
from sklearn.naive_bayes import GaussianNB
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#my code:
#########################################################

clf = GaussianNB()

t0 = time()
#train our classifier with training data and track training time
trained_clf = clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")


#test the accuracy of trained classifier against testing data
y_pred = trained_clf.predict(features_test)


#do a little work to calculate the accuracy score of y_pred
count = 0
i = 0
x = len(y_pred)
while i < x:
    if y_pred[i] == labels_test[i]:
        count += 1
    i += 1

accuracy = float(count) / float(x)
print("trained_clf has an accuracy score of ", accuracy)

#########################################################


