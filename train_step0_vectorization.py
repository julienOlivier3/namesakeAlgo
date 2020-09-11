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
nltk.data.path.append(r'Q:\Meine Bibliotheken\Research\04_Data\03_NLTK')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
import tensorflow_addons as tfa
import keras

import fasttext.util
# -

# # TF-IDF 

# ## Data Cleaning 

df = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_train_d50.pkl")

df.head()

df.shape

# Drop rows with short text data
min_words = 100
row_i = df.clean_text.apply(lambda x: len(x.split()))
df = df.loc[(row_i >= min_words).values,:]

df.shape

# ## Train Test Split & Text Vectorization 

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

newlist = [item for items in temp.y_true_up_bool for item in items]

# Look which technologies are relevant for 'lieken urkorn' (bakery)
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
pd.Series(y.sum(axis=1)).value_counts().sort_index()

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

# + [markdown] heading_collapsed=true
# ## Train-Dev-Test Split & Text Vectorization 

# + hidden=true
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


# + hidden=true
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


# + hidden=true
# Define vectorizer
vectorizer = TfidfVectorizer(
                  encoding = 'utf-8'
                , preprocessor = text_standardization
                , max_df = 0.9
                , min_df = 0.1
                , max_features = 20000
                            )

# + hidden=true
# Train vectorizer
vectorizer.fit(X_train_text)

# + hidden=true
# Transform text data according to trained vectorizer
X_train = vectorizer.transform(X_train_text).toarray().astype(np.float32)
X_dev = vectorizer.transform(X_dev_text).toarray().astype(np.float32)
X_test = vectorizer.transform(X_test_text).toarray().astype(np.float32)

# + hidden=true
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

# + hidden=true
# Calculate number of hot technologies per company
pd.Series(y.sum(axis=1)).value_counts()

# + hidden=true
# Calculate technology distribution
pd.Series(y.sum(axis=0)).plot.bar()

# + hidden=true
# Get all data into one dictionary
dict_train = {'X_train': X_train,
              'X_dev': X_dev,
              'X_test': X_test,
              'y_train': y_train,
              'y_dev': y_dev,
              'y_test': y_test}

# + hidden=true
import sys
sys.getsizeof(dict_train)

# + hidden=true
# Write data
with open(r'Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\dict_train_s0_d50_tfidfPerc.pkl', 'wb') as f:
    pickle.dump(obj=dict_train, file=f)

# + [markdown] heading_collapsed=true
# # Embeddings 

# + [markdown] heading_collapsed=true hidden=true
# ## Webdata

# + hidden=true
df = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_train.pkl")

# + hidden=true
# Drop rows with short text data
min_words = 100
row_i = df.clean_text.apply(lambda x: len(x.split()))
df = df.loc[(row_i >= min_words).values,:]

# + hidden=true
df.head()

# + hidden=true
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


# + hidden=true
# Standardize text
def text_standardization(input_data):
    '''
    lowercase, delete html tags, delte whitespaces, delete numbers, delete punctuation
    
    '''
    
    # lowercasing
    #clean_text = input_data.lower()
    clean_text = input_data
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


# + hidden=true
# Define vectorizer
vectorizer = TfidfVectorizer(
                  encoding = 'utf-8'
                , lowercase = False
                , preprocessor = text_standardization
                , max_df = 0.95
                , min_df = 3
                , max_features = 2000
                #, use_idf = False
                #, tokenizer=nltk.word_tokenize
                            )

# + hidden=true
# Train vectorizer
vectorizer.fit(X_train_text)

# + hidden=true
len(vectorizer.vocabulary_)

# + hidden=true
# Load fastText German Word embeddings
ft = fasttext.load_model(r"H:\Large_Datasets\FastText\cc.de.50.bin")

# + hidden=true
# Dimension of embeddings
ft.get_dimension()

# + hidden=true
# Define vocabulary consisting of most frequent words 
voc = vectorizer.get_feature_names()
word_index = dict(zip(voc, range(0, len(voc))))

# + hidden=true
temp = vectorizer.transform(X_train_text)

# + hidden=true
np.argsort(temp[0].toarray())[0][::-1]

# + hidden=true
temp[0].toarray()[0][555]

# + hidden=true
list(word_index.items())[555]

# + hidden=true
X_train_text[0]

# + hidden=true
ft.get_word_vector('Netzwerk')

# + hidden=true
num_tokens = len(voc) + 1
embedding_dim = ft.get_dimension()
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = ft.get_word_vector(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

# + hidden=true
vocs = np.array(voc)

# + hidden=true
temp2 = temp.toarray()

# + hidden=true
np.argsort(temp2[0])[::-1][:2000]


# + hidden=true
def sentences_to_indices(X, pretrained_model, max_len, vocs, pretrained_vectorizer):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    pretrained_model -- a pretrained word embedding model
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    
    transformed_matrix = pretrained_vectorizer.transform(X).toarray()
    
    m = X.shape[0]                                   # number of training examples
    

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros([m, max_len], dtype = int)
    
    
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence into a list of words.
        #sentence_words = nltk.word_tokenize(X[i])
        #voc_words = np.intersect1d(sentence_words, vocs)
        
        voc_id = np.argsort(transformed_matrix[i])[::-1][:max_len]
        X_indices[i] = voc_id
        #voc_words = vocs[voc_id]
        
        # Reduce size of relevant words to max_len
        #if len(voc_words) > max_len:
        #    voc_words = voc_words[0:max_len]
        
        # Initialize j to 0
        #j = 0
        
        # Loop over the words of sentence_words
        #for w in voc_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            #X_indices[i, j] = int(pretrained_model.get_word_id(w))
            # Increment j to j + 1
            #j = j + 1
        
        if i%1000 == 0:
            print(i)

            
    return X_indices

# + hidden=true
X_train_indices = sentences_to_indices(X_train_text, ft, max_len = 2000, vocs = vocs, pretrained_vectorizer=vectorizer)

# + hidden=true
X_dev_indices = sentences_to_indices(X_dev_text, ft, max_len = 2000, vocs = vocs, pretrained_vectorizer=vectorizer)

# + hidden=true
X_train_indices

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train_indices.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
embedding_layer = tf.keras.layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False
)


# + hidden=true
def model_embedding(input_shape, outputs_dim):
    # Build neural network
    inputs = tf.keras.Input(shape=input_shape, dtype="int32", name='text')
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    #embedding_layer = pretrained_embedding_layer(pretrained_model)
    
    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(inputs)

    x = tf.keras.layers.Flatten()(embeddings)

    #inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

    # Vanilla hidden layer:
    x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Project onto a single unit output layer, and squash it with a sigmoid:
    predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)
    return model


# + hidden=true
model = model_embedding(input_shape=(2000,), outputs_dim=outputs_dim)

# + hidden=true
model.summary()

# + hidden=true
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train_indices
          , y_train
          , validation_data=(X_dev_indices, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          #, callbacks=callbacks_list
                     )

# + hidden=true
model_res.model.evaluate(X_dev_indices, y_dev)


# + hidden=true
# Plot learning performance of NN
def plot_history(history, measure = 'accuracy'):
    acc = history.history[measure]
    val_acc = history.history['val_'+ measure]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training ' + measure)
    plt.plot(x, val_acc, 'r', label='Validation ' + measure)
    plt.title('Training and validation ' + measure)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


# + hidden=true
# Metric to calculate performance of NN
def adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, beta=1, full_res = False, threshold = 0.5):
    # Check whether there is a clear decision by the network and which industry it refers to
    bests = []
    for i in range(y_dev_pred.shape[0]):
        best = list(np.asarray([1 if j>threshold and j==max(y_dev_pred[i]) else 0 for j in y_dev_pred[i]]))
        best = np.argwhere(best == np.amax(best)).flatten()
        bests.append(best)

    bests_train = []
    for i in range(y_train_pred.shape[0]):
        best = list(np.asarray([1 if j>threshold and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
        best = np.argwhere(best == np.amax(best)).flatten()
        bests_train.append(best)
    
    
    # Index of best prediction
    best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
    best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])      

        
    # Drop unpredictable cases
    best_index_clear = best_index[~np.isnan(best_index)]
    y_dev_clear = y_dev[~np.isnan(best_index)]

    best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
    y_train_clear = y_train[~np.isnan(best_train_index)]
    
    
    # Check if best prediction is among true technologies
    trues = []
    for i in range(len(y_dev_clear)):
        true = y_dev_clear[i, int(best_index_clear[i])]
        trues.append(true)

    trues_train = []
    for i in range(len(y_train_clear)):
        true = y_train_clear[i, int(best_train_index_clear[i])]
        trues_train.append(true)
        
    # Calculate metrics
    TP = sum(trues)
    FP = len(trues)-TP 
    FN = np.isnan(best_index).sum()
    no_pred = FN/len(best_index)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F_beta = (1 + beta**2)*(Precision*Recall)/((beta**2 * Precision) + Recall)
    
    TP_train = sum(trues_train)
    FP_train = len(trues_train)-TP_train
    FN_train = np.isnan(best_train_index).sum()
    no_pred_train = FN_train/len(best_train_index)
    Precision_train = TP_train/(TP_train+FP_train)
    Recall_train = TP_train/(TP_train+FN_train)
    F_beta_train = (1 + beta**2)*(Precision_train*Recall_train)/((beta**2 * Precision_train) + Recall_train)
    
    if full_res:
        return print('Validation set precision: ' + '{:.2%}'.format(Precision) + '\n' 
                 + 'Validation set recall: ' + '{:.2%}'.format(Recall) + '\n' 
                 + 'Validation set F-score: ' + '{:.2%}'.format(F_beta) + '\n' 
                 + '\n' +
                 'Training set precision: ' + '{:.2%}'.format(Precision_train) + '\n' 
                 + 'Training set recall: ' + '{:.2%}'.format(Recall_train) + '\n' 
                 + 'Training set F-score: ' + '{:.2%}'.format(F_beta_train))
    else:
        return print('Validation set precision: ' + '{:.2%}'.format(Precision) + '\n' 
                 + 'No prediction made (no prediction above 0.5) for: ' + '{:.2%}'.format(no_pred) + '\n' 
                 + '\n' +
                 'Training set precision: ' + '{:.2%}'.format(Precision_train) + '\n' 
                 + 'No prediction made (no prediction above 0.5) for:  ' + '{:.2%}'.format(no_pred_train))

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev_indices)
y_train_pred = model.predict(X_train_indices)

# + hidden=true
# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, threshold=0.5)

# + [markdown] heading_collapsed=true hidden=true
# ## next
#

# + hidden=true
embedding_layer = tf.keras.layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=True
)


# + hidden=true
def model_embedding(input_shape, outputs_dim):
    # Build neural network
    inputs = tf.keras.Input(shape=input_shape, dtype="int32", name='text')
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    #embedding_layer = pretrained_embedding_layer(pretrained_model)
    
    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(inputs)

    x = tf.keras.layers.Flatten()(embeddings)

    #inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

    # Vanilla hidden layer:
    x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Project onto a single unit output layer, and squash it with a sigmoid:
    predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)
    return model


# + hidden=true
model = model_embedding(input_shape=(2000,), outputs_dim=outputs_dim)

# + hidden=true
model.summary()

# + hidden=true
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train_indices
          , y_train
          , validation_data=(X_dev_indices, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          #, callbacks=callbacks_list
                     )

# + hidden=true
model_res.model.evaluate(X_dev_indices, y_dev)


# + hidden=true
# Plot learning performance of NN
def plot_history(history, measure = 'accuracy'):
    acc = history.history[measure]
    val_acc = history.history['val_'+ measure]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training ' + measure)
    plt.plot(x, val_acc, 'r', label='Validation ' + measure)
    plt.title('Training and validation ' + measure)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


# + hidden=true
# Metric to calculate performance of NN
def adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, beta=1, full_res = False, threshold = 0.5):
    # Check whether there is a clear decision by the network and which industry it refers to
    bests = []
    for i in range(y_dev_pred.shape[0]):
        best = list(np.asarray([1 if j>threshold and j==max(y_dev_pred[i]) else 0 for j in y_dev_pred[i]]))
        best = np.argwhere(best == np.amax(best)).flatten()
        bests.append(best)

    bests_train = []
    for i in range(y_train_pred.shape[0]):
        best = list(np.asarray([1 if j>threshold and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
        best = np.argwhere(best == np.amax(best)).flatten()
        bests_train.append(best)
    
    
    # Index of best prediction
    best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
    best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])      

        
    # Drop unpredictable cases
    best_index_clear = best_index[~np.isnan(best_index)]
    y_dev_clear = y_dev[~np.isnan(best_index)]

    best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
    y_train_clear = y_train[~np.isnan(best_train_index)]
    
    
    # Check if best prediction is among true technologies
    trues = []
    for i in range(len(y_dev_clear)):
        true = y_dev_clear[i, int(best_index_clear[i])]
        trues.append(true)

    trues_train = []
    for i in range(len(y_train_clear)):
        true = y_train_clear[i, int(best_train_index_clear[i])]
        trues_train.append(true)
        
    # Calculate metrics
    TP = sum(trues)
    FP = len(trues)-TP 
    FN = np.isnan(best_index).sum()
    no_pred = FN/len(best_index)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F_beta = (1 + beta**2)*(Precision*Recall)/((beta**2 * Precision) + Recall)
    
    TP_train = sum(trues_train)
    FP_train = len(trues_train)-TP_train
    FN_train = np.isnan(best_train_index).sum()
    no_pred_train = FN_train/len(best_train_index)
    Precision_train = TP_train/(TP_train+FP_train)
    Recall_train = TP_train/(TP_train+FN_train)
    F_beta_train = (1 + beta**2)*(Precision_train*Recall_train)/((beta**2 * Precision_train) + Recall_train)
    
    if full_res:
        return print('Validation set precision: ' + '{:.2%}'.format(Precision) + '\n' 
                 + 'Validation set recall: ' + '{:.2%}'.format(Recall) + '\n' 
                 + 'Validation set F-score: ' + '{:.2%}'.format(F_beta) + '\n' 
                 + '\n' +
                 'Training set precision: ' + '{:.2%}'.format(Precision_train) + '\n' 
                 + 'Training set recall: ' + '{:.2%}'.format(Recall_train) + '\n' 
                 + 'Training set F-score: ' + '{:.2%}'.format(F_beta_train))
    else:
        return print('Validation set precision: ' + '{:.2%}'.format(Precision) + '\n' 
                 + 'No prediction made (no prediction above 0.5) for: ' + '{:.2%}'.format(no_pred) + '\n' 
                 + '\n' +
                 'Training set precision: ' + '{:.2%}'.format(Precision_train) + '\n' 
                 + 'No prediction made (no prediction above 0.5) for:  ' + '{:.2%}'.format(no_pred_train))

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev_indices)
y_train_pred = model.predict(X_train_indices)

# + hidden=true
# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, threshold=0.5)

# + [markdown] heading_collapsed=true hidden=true
# ## next2
#

# + hidden=true
embedding_layer = tf.keras.layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=True
)


# + hidden=true
def model_embedding(input_shape, outputs_dim):
    # Build neural network
    inputs = tf.keras.Input(shape=input_shape, dtype="int32", name='text')
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    #embedding_layer = pretrained_embedding_layer(pretrained_model)
    
    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(inputs)
    x = tf.keras.layers.Conv1D(100, 9, activation="relu")(embeddings)
    x = tf.keras.layers.MaxPooling1D(5)(x)
    x = tf.keras.layers.Conv1D(200, 5, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(5)(x)
    x = tf.keras.layers.Conv1D(400, 3, activation="relu")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    #x = tf.keras.layers.Flatten()(embeddings)

    #inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

    # Vanilla hidden layer:
    x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Project onto a single unit output layer, and squash it with a sigmoid:
    predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)
    return model


# + hidden=true
model = model_embedding(input_shape=(2000,), outputs_dim=outputs_dim)

# + hidden=true
model.summary()

# + hidden=true
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train_indices
          , y_train
          , validation_data=(X_dev_indices, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          #, callbacks=callbacks_list
                     )

# + hidden=true
model_res.model.evaluate(X_dev_indices, y_dev)


# + hidden=true
# Plot learning performance of NN
def plot_history(history, measure = 'accuracy'):
    acc = history.history[measure]
    val_acc = history.history['val_'+ measure]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training ' + measure)
    plt.plot(x, val_acc, 'r', label='Validation ' + measure)
    plt.title('Training and validation ' + measure)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


# + hidden=true
# Metric to calculate performance of NN
def adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, beta=1, full_res = False, threshold = 0.5):
    # Check whether there is a clear decision by the network and which industry it refers to
    bests = []
    for i in range(y_dev_pred.shape[0]):
        best = list(np.asarray([1 if j>threshold and j==max(y_dev_pred[i]) else 0 for j in y_dev_pred[i]]))
        best = np.argwhere(best == np.amax(best)).flatten()
        bests.append(best)

    bests_train = []
    for i in range(y_train_pred.shape[0]):
        best = list(np.asarray([1 if j>threshold and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
        best = np.argwhere(best == np.amax(best)).flatten()
        bests_train.append(best)
    
    
    # Index of best prediction
    best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
    best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])      

        
    # Drop unpredictable cases
    best_index_clear = best_index[~np.isnan(best_index)]
    y_dev_clear = y_dev[~np.isnan(best_index)]

    best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
    y_train_clear = y_train[~np.isnan(best_train_index)]
    
    
    # Check if best prediction is among true technologies
    trues = []
    for i in range(len(y_dev_clear)):
        true = y_dev_clear[i, int(best_index_clear[i])]
        trues.append(true)

    trues_train = []
    for i in range(len(y_train_clear)):
        true = y_train_clear[i, int(best_train_index_clear[i])]
        trues_train.append(true)
        
    # Calculate metrics
    TP = sum(trues)
    FP = len(trues)-TP 
    FN = np.isnan(best_index).sum()
    no_pred = FN/len(best_index)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F_beta = (1 + beta**2)*(Precision*Recall)/((beta**2 * Precision) + Recall)
    
    TP_train = sum(trues_train)
    FP_train = len(trues_train)-TP_train
    FN_train = np.isnan(best_train_index).sum()
    no_pred_train = FN_train/len(best_train_index)
    Precision_train = TP_train/(TP_train+FP_train)
    Recall_train = TP_train/(TP_train+FN_train)
    F_beta_train = (1 + beta**2)*(Precision_train*Recall_train)/((beta**2 * Precision_train) + Recall_train)
    
    if full_res:
        return print('Validation set precision: ' + '{:.2%}'.format(Precision) + '\n' 
                 + 'Validation set recall: ' + '{:.2%}'.format(Recall) + '\n' 
                 + 'Validation set F-score: ' + '{:.2%}'.format(F_beta) + '\n' 
                 + '\n' +
                 'Training set precision: ' + '{:.2%}'.format(Precision_train) + '\n' 
                 + 'Training set recall: ' + '{:.2%}'.format(Recall_train) + '\n' 
                 + 'Training set F-score: ' + '{:.2%}'.format(F_beta_train))
    else:
        return print('Validation set precision: ' + '{:.2%}'.format(Precision) + '\n' 
                 + 'No prediction made (no prediction above 0.5) for: ' + '{:.2%}'.format(no_pred) + '\n' 
                 + '\n' +
                 'Training set precision: ' + '{:.2%}'.format(Precision_train) + '\n' 
                 + 'No prediction made (no prediction above 0.5) for:  ' + '{:.2%}'.format(no_pred_train))

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev_indices)
y_train_pred = model.predict(X_train_indices)

# + hidden=true
# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, threshold=0.5)

# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true
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


# + hidden=true
df['clean_text'] = df.clean_text.apply(lambda x: text_standardization(x))

# + hidden=true
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
# + hidden=true
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
max_len = df.clean_text.apply(lambda x: len(x)).max()
vectorizer_tf = TextVectorization(max_tokens=2000, 
                               output_sequence_length=1000, 
                               standardize=None,
                               #split=nltk.word_tokenize, 
                               ngrams=None, 
                               output_mode='int',
                               pad_to_max_tokens=True)


vectorizer_tf.adapt(X_train_text)

# + hidden=true
vectorizer_tf.get_vocabulary()[:5]

# + hidden=true
dir(vectorizer)

# + hidden=true
vectorizer_tf(np.array([[s] for s in X_train_text[0:3]])).numpy()

# + hidden=true
X_train_indices = vectorizer(np.array([[s] for s in X_train_text])).numpy()
X_dev_indices = vectorizer(np.array([[s] for s in X_dev_text])).numpy()

# + hidden=true
len(voc)

# + hidden=true
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(2, len(voc))))

# + hidden=true
num_tokens = len(voc) + 1
embedding_dim = 50
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = ft.get_word_vector(word.decode("utf-8", errors="ignore"))
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

# + hidden=true
embedding_layer = tf.keras.layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

# + hidden=true
# Build neural network
cdesc_input = tf.keras.Input(shape=input_shape, dtype="int32")
# Create the embedding layer pretrained with GloVe Vectors (≈1 line)
#embedding_layer = pretrained_embedding_layer(pretrained_model)
    
# Propagate sentence_indices through your embedding layer
# (See additional hints in the instructions).
embeddings = embedding_layer(cdesc_input)

x = tf.keras.Flatten(embeddings)

#inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)
# + hidden=true
voc


# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
df.loc[:,'text_token'] = df.clean_text.apply(lambda x: nltk.word_tokenize(x))

# + hidden=true
df.head()

# + [markdown] heading_collapsed=true hidden=true
# ## Company Descriptions 

# + hidden=true
df = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_train_cdesc.pkl")

# + hidden=true
# Tokenization
df.loc[:,'cdesc_token'] = df.cdesc.apply(lambda x: nltk.word_tokenize(x))

# + hidden=true
# No need to drop firms with only few words as descriptions
df.loc[df.cdesc_token.apply(lambda x: len(x))< 3,'cdesc'].value_counts()

# + hidden=true
# Range of number of words for company descriptions
df.cdesc_token.apply(lambda x: len(x)).min(), df.cdesc_token.apply(lambda x: len(x)).max()

# + hidden=true
# Distribution of number of words for company descriptions
df.cdesc_token.apply(lambda x: len(x)).value_counts().plot.hist(bins=60)

# + hidden=true
# Load fastText German Word embeddings
ft = fasttext.load_model(r"H:\Large_Datasets\FastText\cc.de.50.bin")

# + hidden=true active=""
# # Reduce model size
# ft = fasttext.load_model(r"H:\Large_Datasets\FastText\cc.de.50.bin")
# fasttext.util.reduce_model(ft, X)
#
# ft.save_model("cc.de.X.bin")

# + hidden=true
# Dimension of embeddings
ft.get_dimension()

# + hidden=true
# Number of words for which pretrained embeddings exist
len(ft.get_labels())

# + hidden=true
# Word to index
ft.get_word_id('Müllverbrennungsanlage')

# + hidden=true
# Index to word
ft.get_words()[137440], ft.get_labels()[137440]

# + hidden=true
# Word to embedding layer
ft.get_word_vector('Müllverbrennungsanlage')

# + hidden=true
# Nearest neighbors given word
ft.get_nearest_neighbors('Holdinggesellschaft')


# + [markdown] heading_collapsed=true hidden=true
# ## Manual Approach 

# + hidden=true
def sentences_to_indices(X, pretrained_model, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    pretrained_model -- a pretrained word embedding model
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros([m, max_len])
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence into a list of words.
        sentence_words = nltk.word_tokenize(X[i])
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = int(pretrained_model.get_word_id(w))
            # Increment j to j + 1
            j = j + 1
            
    
    return X_indices

# + hidden=true
X1 = np.array(["Gartencenter", "Biogasanlage zur Herstellung alternativer Energie", "Müllverbrennungsanlage"])
X1_indices = sentences_to_indices(X1, pretrained_model = ft, max_len = 5)
print("X1 =", X1)
print("X1_indices =\n", X1_indices)

# + hidden=true
len(ft.get_labels())


# + hidden=true
def pretrained_embedding_layer(pretrained_model):
    """
    Creates a Keras Embedding() layer and loads in pre-trained fastText 50-dimensional vectors (German).
    
    Arguments:
    pretrained_model -- a pretrained word embedding model

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(pretrained_model.get_labels()) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = pretrained_model.get_dimension()      # define dimensionality of your fastText word vectors (= 50)
    
    # Step 1
    # Initialize the embedding matrix as a numpy array of zeros.
    # See instructions above to choose the correct shape.
    emb_matrix = np.zeros([vocab_len, emb_dim])
    
    # Step 2
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word in pretrained_model.get_words():
        emb_matrix[pretrained_model.get_word_id(word), :] = pretrained_model.get_word_vector(word)

    # Step 3
    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = tf.keras.layers.Embedding(
        input_dim = vocab_len, 
        output_dim = emb_dim, 
        #input_length = max_len,
        trainable=False,
        #embeddings_initializer=tf.keras.initializers.Constant(emb_matrix)
    )
    ### END CODE HERE ###

    # Step 4 (already done for you; please do not modify)
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


# + hidden=true
def model_embedding(input_shape, outputs_dim, pretrained_model):
    cdesc_input = tf.keras.Input(shape=input_shape, dtype="int32")
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(pretrained_model)
    
    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(cdesc_input)   

    x = tf.keras.layers.Conv1D(128, 5, activation="relu")(embeddings)
    x = tf.keras.layers.MaxPooling1D(5)(x)
    x = tf.keras.layers.Conv1D(128, 5, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(5)(x)
    x = tf.keras.layers.Conv1D(128, 5, activation="relu")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Project onto a single unit output layer, and squash it with a sigmoid:
    predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(cdesc_input, predictions)
    
    
    return model

# + hidden=true
embedding_layer = pretrained_embedding_layer(ft)

# + hidden=true
embedding_layer.shape

# + hidden=true
print("weights[0] =", embedding_layer.get_weights()[0])

# + hidden=true
X_train_indices = sentences_to_indices(X_train_text, pretrained_model = ft, max_len = 2000)
X_dev_indices = sentences_to_indices(X_dev_text, pretrained_model = ft, max_len = 2000)

# + [markdown] heading_collapsed=true hidden=true
# ## Keras Approach 

# + hidden=true
# Train-dev-test split
X = df.cdesc.values
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

# + hidden=true
X_train_text

# + hidden=true
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
max_len = df.cdesc_token.apply(lambda x: len(x)).max()
max_len

# + hidden=true
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
max_len = df.cdesc_token.apply(lambda x: len(x)).max()
vectorizer = TextVectorization(max_tokens=None, 
                               output_sequence_length=int(max_len), 
                               standardize=None,
                               #split=nltk.word_tokenize, 
                               ngrams=None, 
                               output_mode='int',
                               pad_to_max_tokens=True)


vectorizer.adapt(X_train_text)

# + hidden=true
vectorizer.get_vocabulary()[:5]

# + hidden=true
output = vectorizer(np.array([["und die Müllverbrennungsanlage liegt an der Straße"]]))
output.numpy()[0, :6]

# + hidden=true
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(2, len(voc))))

# + hidden=true
len(voc)

# + hidden=true
voc[0]


# + hidden=true
ft.get_word_vector(b'und')

# + hidden=true
num_tokens = len(voc) + 2
embedding_dim = 50
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = ft.get_word_vector(word.decode("utf-8", errors="ignore"))
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

# + hidden=true
ft.get_word_vector(b'",')

# + hidden=true
embedding_matrix.shape

# + hidden=true
embedding_matrix[0], embedding_matrix[1], embedding_matrix[113789-1], embedding_matrix[113789], embedding_matrix[embedding_matrix.shape[0]-1]

# + hidden=true
embedding_layer = tf.keras.layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)


# + hidden=true
def model_embedding(input_shape, outputs_dim, pretrained_model):
    cdesc_input = tf.keras.Input(shape=input_shape, dtype="int32")
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    #embedding_layer = pretrained_embedding_layer(pretrained_model)
    
    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(cdesc_input)   

    x = tf.keras.layers.Conv1D(128, 5, activation="relu")(embeddings)
    x = tf.keras.layers.MaxPooling1D(5)(x)
    x = tf.keras.layers.Conv1D(128, 5, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(5)(x)
    x = tf.keras.layers.Conv1D(128, 5, activation="relu")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Project onto a single unit output layer, and squash it with a sigmoid:
    predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(cdesc_input, predictions)
    
    
    return model

# + hidden=true
max_len = df.cdesc_token.apply(lambda x: len(x)).max() # maximum number of words occuring in company descriptions
#max_len = 150
outputs_dim = len(y_train[0]) # number of output neurons (classes to predict)

model = model_embedding(input_shape = (None,), outputs_dim = outputs_dim, pretrained_model=ft)
model.summary()

# + hidden=true
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )

# + hidden=true
# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights3.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# + hidden=true
X_train_indices = vectorizer(np.array([[s] for s in X_train_text])).numpy()
X_dev_indices = vectorizer(np.array([[s] for s in X_dev_text])).numpy()


# + hidden=true
X_train_indices.shape

# + hidden=true
X_train_indices[15], X_train_text[15] 

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train_indices
          , y_train
          , validation_data=(X_dev_indices, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          #, callbacks=callbacks_list
                     )

# + hidden=true
model_res.model.evaluate(X_dev_indices, y_dev)


# + hidden=true
# Plot learning performance of NN
def plot_history(history, measure = 'accuracy'):
    acc = history.history[measure]
    val_acc = history.history['val_'+ measure]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training ' + measure)
    plt.plot(x, val_acc, 'r', label='Validation ' + measure)
    plt.title('Training and validation ' + measure)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


# + hidden=true
# Metric to calculate performance of NN
def adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, beta=1, full_res = False, threshold = 0.5):
    # Check whether there is a clear decision by the network and which industry it refers to
    bests = []
    for i in range(y_dev_pred.shape[0]):
        best = list(np.asarray([1 if j>threshold and j==max(y_dev_pred[i]) else 0 for j in y_dev_pred[i]]))
        best = np.argwhere(best == np.amax(best)).flatten()
        bests.append(best)

    bests_train = []
    for i in range(y_train_pred.shape[0]):
        best = list(np.asarray([1 if j>threshold and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
        best = np.argwhere(best == np.amax(best)).flatten()
        bests_train.append(best)
    
    
    # Index of best prediction
    best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
    best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])      

        
    # Drop unpredictable cases
    best_index_clear = best_index[~np.isnan(best_index)]
    y_dev_clear = y_dev[~np.isnan(best_index)]

    best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
    y_train_clear = y_train[~np.isnan(best_train_index)]
    
    
    # Check if best prediction is among true technologies
    trues = []
    for i in range(len(y_dev_clear)):
        true = y_dev_clear[i, int(best_index_clear[i])]
        trues.append(true)

    trues_train = []
    for i in range(len(y_train_clear)):
        true = y_train_clear[i, int(best_train_index_clear[i])]
        trues_train.append(true)
        
    # Calculate metrics
    TP = sum(trues)
    FP = len(trues)-TP 
    FN = np.isnan(best_index).sum()
    no_pred = FN/len(best_index)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F_beta = (1 + beta**2)*(Precision*Recall)/((beta**2 * Precision) + Recall)
    
    TP_train = sum(trues_train)
    FP_train = len(trues_train)-TP_train
    FN_train = np.isnan(best_train_index).sum()
    no_pred_train = FN_train/len(best_train_index)
    Precision_train = TP_train/(TP_train+FP_train)
    Recall_train = TP_train/(TP_train+FN_train)
    F_beta_train = (1 + beta**2)*(Precision_train*Recall_train)/((beta**2 * Precision_train) + Recall_train)
    
    if full_res:
        return print('Validation set precision: ' + '{:.2%}'.format(Precision) + '\n' 
                 + 'Validation set recall: ' + '{:.2%}'.format(Recall) + '\n' 
                 + 'Validation set F-score: ' + '{:.2%}'.format(F_beta) + '\n' 
                 + '\n' +
                 'Training set precision: ' + '{:.2%}'.format(Precision_train) + '\n' 
                 + 'Training set recall: ' + '{:.2%}'.format(Recall_train) + '\n' 
                 + 'Training set F-score: ' + '{:.2%}'.format(F_beta_train))
    else:
        return print('Validation set precision: ' + '{:.2%}'.format(Precision) + '\n' 
                 + 'No prediction made (no prediction above 0.5) for: ' + '{:.2%}'.format(no_pred) + '\n' 
                 + '\n' +
                 'Training set precision: ' + '{:.2%}'.format(Precision_train) + '\n' 
                 + 'No prediction made (no prediction above 0.5) for:  ' + '{:.2%}'.format(no_pred_train))

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev_indices)
y_train_pred = model.predict(X_train_indices)

# + hidden=true
# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, threshold=0.5)

# + hidden=true
max_len, outputs_dim
