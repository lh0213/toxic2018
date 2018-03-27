'''
Created on 27 Mar 2018

@author: zhi liang
'''

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.classification import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics.ranking import roc_curve, auc

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

all_Comments_Data = pd.read_csv('train.csv').fillna(' ') #
#test = pd.read_csv('test.csv').fillna(' ')

all_Comments = all_Comments_Data['comment_text']
#test_text = test['comment_text']
#all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=10000)
#word_vectorizer.fit(all_Comments)
#word_features = word_vectorizer.transform(all_Comments)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
#char_vectorizer.fit(all_Comments)
#char_features = char_vectorizer.transform(all_Comments)


scores = []

for class_name in class_names:

    classification = all_Comments_Data[class_name]
    logistic_Classifier = LogisticRegression(C=0.1, solver='sag')
    
    X_train, X_test, y_train, y_test = train_test_split(all_Comments, classification, test_size = 0.01, random_state = 100)
    X_train, X_test_dummy, y_train, y_test_dummy = train_test_split(X_train, y_train, test_size = 0.5, random_state = 100)
    


    X_train = char_vectorizer.fit_transform(X_train)
    X_test  = char_vectorizer.transform(X_test)

    logistic_Classifier.fit(X_train, y_train)
    
    y_pred = logistic_Classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    
    '''
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    '''
    