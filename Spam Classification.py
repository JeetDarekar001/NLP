# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 19:43:56 2022

@author: Jeetr
"""
import pandas as pd

messages = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

wm=WordNetLemmatizer()
ps = PorterStemmer()
corpus = []
wmcorpus=[]
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    wmCor= [wm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    wmcorpus.append(" ".join(wmCor))
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB()
spam_detect_model.fit(X_train, y_train)dcsad

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score


print(accuracy_score(y_test,y_pred))
# Accuracy is  0.9856502242152466

# This Accuracy is with respect to bag of words and Stemming
print(confusion_matrix(y_test,y_pred))
'''
[[946   9]
 [  7 153]]
'''

# Testing with Lemmatizer and tfidf

 from sklearn.feature_extraction.text import TfidfVectorizer
 tf = TfidfVectorizer()
 X_lem = tf.fit_transform(corpus).toarray()

 y=pd.get_dummies(messages['label'])
 y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_lem, y, test_size = 0.20, random_state = 0)
spam_detect_model.fit(X_train,y_train)
y_pred=spam_detect_model.predict(X_test)
print(accuracy_score(y_test,y_pred))
# Accuracy is  0.9695067264573991

# This Accuracy is with respect to bag of words and Stemming
print(confusion_matrix(y_test,y_pred))

'''
[[955   0]
 [ 34 126]]'''