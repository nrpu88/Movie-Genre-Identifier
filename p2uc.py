import sklearn
from sklearn.datasets import load_files
import sys

from os import listdir
import itertools
from sklearn.feature_extraction.text import HashingVectorizer
import csv


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier


from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier



from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords

from sklearn.model_selection import cross_val_score
import decimal
import string

# loading all files as training data. 
movie_dir= r'movie_reviews' 

array = []

movie_train = load_files(movie_dir, shuffle=True)

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens


vectorizer = HashingVectorizer(n_features = 6500,norm=None,ngram_range=(1,2),alternate_sign=False)

vector = vectorizer.transform(movie_train.data)

c = 0
d = 0

docs_train, docs_test, y_train, y_test = train_test_split(
    vector, movie_train.target, test_size = 0.01, random_state = 1)

print(movie_train.target)
print(type(movie_train.target))

for i in movie_train.target:
	if i == 1:
		c = c+1
	else:
		d = d+1

print c
print d
