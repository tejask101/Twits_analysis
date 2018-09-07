# -*- coding: utf-8 -*-
"""
Created on Sat Sep 01 13:45:49 2018

@author: karal
"""
# IIT Ghuwahati lingupedia Natural Language Processing with maximum features of 15000


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset_train = pd.read_csv('train.csv')
dataset_test=pd.read_csv('test.csv')
y_train=dataset_train.iloc[:,1].values


X=np.concatenate((dataset_train['tweet'],dataset_test['tweet']))
temp=pd.DataFrame(data=X)

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 9873):
    review = re.sub('[^a-zA-Z]', ' ', X[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000)
X = cv.fit_transform(corpus).toarray()

#seperating train and test after processing
X_train=X[:7920]
X_test=X[7920:]

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

sub=pd.read_csv('sample_submission.csv')
sub['label']=y_pred
sub.to_csv('submission6.csv',index=False)





