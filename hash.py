import sklearn

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split

import pandas as pd 
import numpy as np

import math

import csv

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier



from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score


from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsClassifier  
import parfit.parfit as pf

import decimal

import time

import warnings
warnings.filterwarnings("ignore")

import sys

orig_stdout = sys.stdout
f = open('stat.txt', 'w')
sys.stdout = f






list2 = []
genre = []



inputfile = csv.reader(open('otest(nofiction).csv','r'))

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

vectorizer = HashingVectorizer(n_features = 20000,ngram_range=(1,1),alternate_sign = False)

vdata = vectorizer.transform(list2)

end = time.time()


print("\nModel construction time:")
print(end - start)
print('\n')

'''

print(np_genre[0:10])
print(len(np_genre))

print(vdata.shape)
print(np_genre.shape)
'''

docs_train, docs_test, y_train, y_test = train_test_split(
    vdata, np_genre, test_size = 0.01, random_state = 1)



dtc = DecisionTreeClassifier(class_weight = 'balanced').fit(docs_train,y_train)

lr = LogisticRegression(solver = 'newton-cg',class_weight = 'balanced',max_iter = 10000).fit(docs_train,y_train)

svm = SVC().fit(docs_train,y_train)

sgd = SGDClassifier(tol = 1e-3,shuffle=False,max_iter = 2000,validation_fraction = 0.01).fit(docs_train, y_train)#the second best

classifier = KNeighborsClassifier(n_neighbors=5).fit(docs_train, y_train)

print("MacroPrecision:")
y_pred = dtc.predict(docs_test)
print("dtc\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'macro')))

print('\n')

y_pred = lr.predict(docs_test)
print("lr\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'macro')))
print("LRMicro")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'micro')))

print('\n')

y_pred = svm.predict(docs_test)
print("svm\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'macro')))

y_pred = sgd.predict(docs_test)
print("sgd\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'macro')))
print('SGDMICRO:')
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'micro')))
print('SGDacc:')
print(decimal.Decimal(sklearn.metrics.accuracy_score(y_test, y_pred)))

y_pred = classifier.predict(docs_test)
print("KNeighborsClassifier\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'macro')))
print("\nMacroRecall:")
y_pred = dtc.predict(docs_test)
print("dtc\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'macro')))

print('\n')
y_pred = lr.predict(docs_test)
print("lr\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'macro')))
print('\n')

y_pred = svm.predict(docs_test)
print("svm\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'macro')))

y_pred = sgd.predict(docs_test)
print("sgd\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'macro')))


y_pred = classifier.predict(docs_test)
print("KNeighborsClassifier\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'macro')))

print("\nMicroRecall:")	
y_pred = dtc.predict(docs_test)
print("dtc\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'micro')))

print('\n')
y_pred = lr.predict(docs_test)
print("lr\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'micro')))
print('\n')

y_pred = svm.predict(docs_test)
print("svm\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'micro')))

y_pred = sgd.predict(docs_test)
print("sgd\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'micro')))


y_pred = classifier.predict(docs_test)
print("KNeighborsClassifier\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'micro')))

print("\nMicroF1:")
y_pred = dtc.predict(docs_test)
print("dtc\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'micro')))

print('\n')
y_pred = lr.predict(docs_test)
print("lr\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'micro')))
print('\n')

y_pred = svm.predict(docs_test)
print("svm\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'micro')))

y_pred = sgd.predict(docs_test)
print("sgd\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'micro')))


y_pred = classifier.predict(docs_test)
print("KNeighborsClassifier\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'micro')))

print("\nMacroF1:")
y_pred = dtc.predict(docs_test)
print("dtc\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'macro')))

print('\n')
y_pred = lr.predict(docs_test)
print("lr\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'macro')))
print('\n')

y_pred = svm.predict(docs_test)
print("svm\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'macro')))

y_pred = sgd.predict(docs_test)
print("sgd\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'macro')))


y_pred = classifier.predict(docs_test)
print("KNeighborsClassifier\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'macro')))
print("\ndone")


print("\nWeighted Param(Only LR):")
print('F1:\n')
y_pred = lr.predict(docs_test)
print("lr\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'weighted')))
print('\n')

print('Recall:\n')
y_pred = lr.predict(docs_test)
print("lr\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'weighted')))
print('\n')

print('Precision:\n')
y_pred = lr.predict(docs_test)
print("lr\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'weighted')))
print('\n')



print("\nWeighted Param(Only SGD):")
print('F1:\n')
y_pred = sgd.predict(docs_test)
print("sgd\t")
print(decimal.Decimal(sklearn.metrics.f1_score(y_test, y_pred,average = 'weighted')))
print('\n')

print('Recall:\n')
y_pred = sgd.predict(docs_test)
print("sgd\t")
print(decimal.Decimal(sklearn.metrics.recall_score(y_test, y_pred,average = 'weighted')))
print('\n')

print('Precision:\n')
y_pred = sgd.predict(docs_test)
print("sgd\t")
print(decimal.Decimal(sklearn.metrics.precision_score(y_test, y_pred,average = 'weighted')))
print('\n')


sys.stdout = orig_stdout
f.close()
