# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import pickle
import re
import string
import pydot
import math
import matplotlib.pyplot as plt

import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# -

# # Data Cleaning 

df = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_train.pkl")

df.head()

# Drop rows with short text data
min_words = 100
row_i = df.clean_text.apply(lambda x: len(x.split()))
df = df.loc[(row_i >= min_words).values,:]

# # Train Test Split & Text Vectorization 

# +
# Train test split
X = df.clean_text.values
y = df.y_true_up_bool.values

# Convert target as clean numpy array
n = y.shape[0]
a = []
for i in range(n):
    a.append([j for j in y[i]])
y = np.array(a)

X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=333)
# -

X_test_text[0]

temp=df.loc[df.clean_text.apply(lambda x: re.search('lieken urkorn', x.lower()) is not None),:]

newlist = [item for items in temp.y_true_bool for item in items]

for index, i in enumerate(newlist):
    if i == 1:
        print(index+1)


# Standardize text
def text_standardization(input_data):
    '''
    lowercase, delete html tags, delte whitespaces, delete numbers, delete punctuation
    
    '''
    
    # lowercasing
    clean_text = input_data.lower()
    # delete html tags
    clean_text = re.sub(r'\[->.*?<-\]', ' ', clean_text)
    # delete leading and trailing whitespaces
    clean_text = clean_text.strip()
    clean_text = re.sub(r' +', ' ', clean_text)
    # delete numbers not part of a word
    clean_text = re.sub(r'\b\d+\b', '', clean_text)
    # delete punctuation
    clean_text = re.sub("[%s]" % re.escape(string.punctuation), "", clean_text)
    return clean_text


# Define vectorizer
vectorizer = TfidfVectorizer(
                  encoding = 'utf-8'
                , preprocessor = text_standardization
                , max_df = 0.95
                , min_df = 3
                , max_features = 20000
                            )

# Train vectorizer
vectorizer.fit(X_train_text)

# Transform text data according to trained vectorizer
X_train = vectorizer.transform(X_train_text).toarray().astype(np.float32)
X_test = vectorizer.transform(X_test_text).toarray().astype(np.float32)

# +
# Get relevant word indices from first webpage in trainings data
indexes = []
for index, i in enumerate(X_train[0]):
    if i!=0:
        indexes.append(index)
indexes

# Extract respetcive words from vectorized vocabulary list
voc_dict = {y:x for x,y in vectorizer.vocabulary_.items()}

words = []
for i in indexes:
    word = voc_dict[i]
    words.append(word)
words

# Check if words can be found in the text
word_matches = [re.search(w, X_train_text[0].lower()).group(0) for w in words if re.search(w, X_train_text[0].lower())]
len(word_matches), len(words)

# Look at non-matches
np.setdiff1d(words,word_matches)
# after removing punctuations these three words can be found as well -> all fine
# -

# Calculate number of hot technologies per company
pd.Series(y.sum(axis=1)).value_counts()

# Calculate technology distribution
pd.Series(y.sum(axis=0)).plot.bar()

# Get all data into one dictionary
dict_train = {'X_train': X_train,
             'X_test': X_test,
             'y_train': y_train,
             'y_test': y_test}

import sys
sys.getsizeof(dict_train)

# Write data
with open(r'Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\dict_train_s0.pkl', 'wb') as f:
    pickle.dump(obj=dict_train, file=f)

# # Train-Dev-Test Split & Text Vectorization 

# +
# Train-dev-test split
X = df.clean_text.values
y = df.y_true_up_bool.values

# Convert target as clean numpy array
n = y.shape[0]
a = []
for i in range(n):
    a.append([j for j in y[i]])
y = np.array(a)

X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=333)

X_dev_text, X_test_text, y_dev, y_test = train_test_split(X_test_text, y_test, test_size = 0.5, random_state=333)

print(str(X_train_text.shape[0]) + ' is the number of training samples.')
print(str(X_dev_text.shape[0]) + ' is the number of development/validation samples.')
print(str(X_test_text.shape[0]) + ' is the number of idependent test samples.')


# -

# Standardize text
def text_standardization(input_data):
    '''
    lowercase, delete html tags, delte whitespaces, delete numbers, delete punctuation
    
    '''
    
    # lowercasing
    clean_text = input_data.lower()
    # delete html tags
    clean_text = re.sub(r'\[->.*?<-\]', ' ', clean_text)
    # delete leading and trailing whitespaces
    clean_text = clean_text.strip()
    clean_text = re.sub(r' +', ' ', clean_text)
    # delete numbers not part of a word
    clean_text = re.sub(r'\b\d+\b', '', clean_text)
    # delete punctuation
    clean_text = re.sub("[%s]" % re.escape(string.punctuation), "", clean_text)
    return clean_text


# Define vectorizer
vectorizer = TfidfVectorizer(
                  encoding = 'utf-8'
                , preprocessor = text_standardization
                , max_df = 0.95
                , min_df = 3
                , max_features = 20000
                            )

# Train vectorizer
vectorizer.fit(X_train_text)

# Transform text data according to trained vectorizer
X_train = vectorizer.transform(X_train_text).toarray().astype(np.float32)
X_dev = vectorizer.transform(X_dev_text).toarray().astype(np.float32)
X_test = vectorizer.transform(X_test_text).toarray().astype(np.float32)

# +
# Get relevant word indices from first webpage in trainings data
indexes = []
for index, i in enumerate(X_train[0]):
    if i!=0:
        indexes.append(index)
indexes

# Extract respetcive words from vectorized vocabulary list
voc_dict = {y:x for x,y in vectorizer.vocabulary_.items()}

words = []
for i in indexes:
    word = voc_dict[i]
    words.append(word)
words

# Check if words can be found in the text
word_matches = [re.search(w, X_train_text[0].lower()).group(0) for w in words if re.search(w, X_train_text[0].lower())]
len(word_matches), len(words)

# Look at non-matches
np.setdiff1d(words,word_matches)
# after removing punctuations these three words can be found as well -> all fine
# -

# Calculate number of hot technologies per company
pd.Series(y.sum(axis=1)).value_counts()

# Calculate technology distribution
pd.Series(y.sum(axis=0)).plot.bar()

# Get all data into one dictionary
dict_train = {'X_train': X_train,
              'X_dev': X_dev,
              'X_test': X_test,
              'y_train': y_train,
              'y_dev': y_dev,
              'y_test': y_test}

import sys
sys.getsizeof(dict_train)

# Write data
with open(r'Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\dict_train_s0.pkl', 'wb') as f:
    pickle.dump(obj=dict_train, file=f)
