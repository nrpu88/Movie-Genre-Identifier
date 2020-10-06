import sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


import pandas as pd 
import numpy as np

import math

import csv

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR

import decimal

import time

import warnings
warnings.filterwarnings("ignore")


list2 = []
genre = []



inputfile = csv.reader(open('otest.csv','r'))

f1 = open("tgenre.txt","w")


for row in inputfile:
	f1.write(row[3])

	genre.append(int(row[3]))
	np_genre = np.array(genre)

	vector2 = row[2]
	vector2 = str(vector2)

	list2.append(vector2)
			
f = open("t1.txt","w")

f.write(str(list2))

start = time.time()

vectorizer = CountVectorizer()

vectorizer.fit(list2)

vdata = vectorizer.transform(list2)

end = time.time()

start1 = time.time()
vectorizer1 = TfidfVectorizer()
vectorizer1.fit(list2)
vdata1 = vectorizer1.transform(list2)
end1 = time.time()


print(end1 - start1)
print(end - start)


docs_train, docs_test, y_train, y_test = train_test_split(
    vdata, np_genre, test_size = 0.01, random_state = 1)

docs_train1, docs_test1, y_train1, y_test1 = train_test_split(
    vdata1, np_genre, test_size = 0.01, random_state = 1)


dtc = DecisionTreeClassifier().fit(docs_train,y_train)
lr = LogisticRegression().fit(docs_train,y_train)
lr1 = LogisticRegression().fit(docs_train1,y_train1)

svm = SVC().fit(docs_train,y_train)

sgd = SGDClassifier(tol = 1e-3,shuffle=False,max_iter = 4000,validation_fraction = 0.01).fit(docs_train, y_train)
sgd1 = SGDClassifier(tol = 1e-3,shuffle=False,max_iter = 4000,validation_fraction = 0.01).fit(docs_train1, y_train1)



y_pred = lr.predict(docs_test)
print("lr\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'micro')))


print("\nSgdCount")
y_pred = sgd.predict(docs_test)
print("sgdMacPrecision\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'macro')))
print('SGDMICROPrecision:')
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'micro')))
print('SGDMacroRecall:')
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'macro')))
print('SGDMicroRecall:')
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'micro')))
print('SGDMacroF1:')
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'macro')))
print('SGDMicroF1:')
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'micro')))


print("\nLRCount")
y_pred = lr.predict(docs_test)
print("lrMacPrecision\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'macro')))
print('lrMICROPrecision:')
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'micro')))
print('lrMacroRecall:')
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'macro')))
print('lrMicroRecall:')
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'micro')))
print('lrMacroF1:')
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'macro')))
print('lrMicroF1:')
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'micro')))

print("\nSgdTfidf")
y_pred1 = sgd1.predict(docs_test1)
print("sgdMacPrecision\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test1, y_pred1,average = 'macro')))
print('SGDMICROPrecision:')
print(decimal.Decimal(sklearn.metrics.precision_score(y_test1, y_pred1,average = 'micro')))
print('SGDMacroRecall:')
print(decimal.Decimal(sklearn.metrics.recall_score(y_test1, y_pred1,average = 'macro')))
print('SGDMicroRecall:')
print(decimal.Decimal(sklearn.metrics.recall_score(y_test1, y_pred1,average = 'micro')))
print('SGDMacroF1:')
print(decimal.Decimal(sklearn.metrics.f1_score(y_test1, y_pred1,average = 'macro')))
print('SGDMicroF1:')
print(decimal.Decimal(sklearn.metrics.f1_score(y_test1, y_pred1,average = 'micro')))

print("\nLRTfidf")
y_pred1 = lr1.predict(docs_test1)
print("lrMacPrecision\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test1, y_pred1,average = 'macro')))
print('lrMICROPrecision:')
print(decimal.Decimal(sklearn.metrics.precision_score(y_test1, y_pred1,average = 'micro')))
print('lrMacroRecall:')
print(decimal.Decimal(sklearn.metrics.recall_score(y_test1, y_pred1,average = 'macro')))
print('lrMicroRecall:')
print(decimal.Decimal(sklearn.metrics.recall_score(y_test1, y_pred1,average = 'micro')))
print('lrMacroF1:')
print(decimal.Decimal(sklearn.metrics.f1_score(y_test1, y_pred1,average = 'macro')))
print('lrMicroF1:')
print(decimal.Decimal(sklearn.metrics.f1_score(y_test1, y_pred1,average = 'micro')))

print("\ndone")

