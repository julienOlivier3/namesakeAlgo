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
import os
import IPython
import random

import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
import tensorflow_addons as tfa
import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import kerastuner as kt
# -

df = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\dict_train_dev_test_s0_d50.pkl")

X_train = df['X_train']
X_dev = df['X_dev']
X_test = df['X_test']
y_train = df['y_train']
y_dev = df['y_dev']
y_test = df['y_test']

# Note that rows are normalized ...
np.min(X_train), np.max(X_train)

# such that the sum of squares of each row (= document) equals 1
np.sum(X_train[0]**2)

# # Functions

# +
# Seed value
# Apparently you may use different seed values at each stage
seed_value= 333

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# +
# Weighted binary cross entropy
import keras.backend.tensorflow_backend as tfb

POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned

def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)


# -

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


# Read word embedding vector
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            try:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            except:
                pass
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


# # Models 

# + [markdown] heading_collapsed=true
# ## 1. Iteration 

# + [markdown] hidden=true
# - __Input__: Top normed technology group counts
# - __Model__: ANN, 2 hidden layers, 128 neurons, multi-label
# - __Loss__: Weighted binary cross entropy

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu")(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Plot the network
#tf.keras.utils.plot_model(model)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss=weighted_binary_crossentropy, 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold = 0.5), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_test, y_test)
          , batch_size=32
          , epochs=epochs)

# + hidden=true
plot_history(model_res, measure = 'precision')

# + hidden=true
# Evaluate model on test set
model.evaluate(X_test, y_test)

# + hidden=true
# Calculate predictions
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# + hidden=true
# Check whether there is a clear decision by the network and which industry it refers to
bests = []
for i in range(y_test_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_test_pred[i]) else 0 for j in y_test_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests.append(best)

pd.Series(bests).apply(len).value_counts()

bests_train = []
for i in range(y_train_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests_train.append(best)

pd.Series(bests_train).apply(len).value_counts()

# + hidden=true
# Index of best prediction
best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])

# + hidden=true
# Drop unpredictable cases
best_index_clear = best_index[~np.isnan(best_index)]
y_test_clear = y_test[~np.isnan(best_index)]

best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
y_train_clear = y_train[~np.isnan(best_train_index)]


# + hidden=true
# Check if best prediction is among true technologies
trues = []
for i in range(len(y_test_clear)):
    true = y_test_clear[i, int(best_index_clear[i])]
    trues.append(true)

trues_train = []
for i in range(len(y_train_clear)):
    true = y_train_clear[i, int(best_train_index_clear[i])]
    trues_train.append(true)

# + hidden=true
# Final result
print('{:.0%}'.format(sum(trues)/len(trues)) + ' of top technology prediction on test data is indeed among the top technologies.')
print('{:.0%}'.format(sum(trues_train)/len(trues_train)) + ' of top technology prediction on trainig data is indeed among the top technologies.')

# + [markdown] heading_collapsed=true
# ## 2. Iteration 

# + [markdown] hidden=true
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 2 hidden layers, 50 neurons, multi-label
# - __Loss__: Weighted binary cross entropy

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(50, activation="relu")(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Vanilla hidden layer:
x = tf.keras.layers.Dense(50, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Plot the network
#tf.keras.utils.plot_model(model)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss=weighted_binary_crossentropy, 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold = 0.5), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_test, y_test)
          , batch_size=32
          , epochs=epochs)

# + hidden=true
plot_history(model_res, measure = 'precision_1')

# + hidden=true
# Evaluate model on test set
model.evaluate(X_test, y_test)

# + hidden=true
# Calculate predictions
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# + hidden=true
# Check whether there is a clear decision by the network and which industry it refers to
bests = []
for i in range(y_test_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_test_pred[i]) else 0 for j in y_test_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests.append(best)

pd.Series(bests).apply(len).value_counts()

bests_train = []
for i in range(y_train_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests_train.append(best)

pd.Series(bests_train).apply(len).value_counts()

# + hidden=true
# Index of best prediction
best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])

# + hidden=true
# Drop unpredictable cases
best_index_clear = best_index[~np.isnan(best_index)]
y_test_clear = y_test[~np.isnan(best_index)]

best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
y_train_clear = y_train[~np.isnan(best_train_index)]


# + hidden=true
# Check if best prediction is among true technologies
trues = []
for i in range(len(y_test_clear)):
    true = y_test_clear[i, int(best_index_clear[i])]
    trues.append(true)

trues_train = []
for i in range(len(y_train_clear)):
    true = y_train_clear[i, int(best_train_index_clear[i])]
    trues_train.append(true)

# + hidden=true
# Final result
print('{:.0%}'.format(sum(trues)/len(trues)) + ' of top technology prediction on test data is indeed among the top technologies.')
print('{:.0%}'.format(sum(trues_train)/len(trues_train)) + ' of top technology prediction on trainig data is indeed among the top technologies.')

# + [markdown] hidden=true
# Worse!

# + [markdown] heading_collapsed=true
# ## 3. Iteration 

# + [markdown] hidden=true
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 2 hidden layers, 128 neurons, multi-label
# - __Loss__: Binary cross entropy

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu")(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Plot the network
#tf.keras.utils.plot_model(model)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold = 0.5), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_test, y_test)
          , batch_size=32
          , epochs=epochs
          , verbose=0)

# + hidden=true
plot_history(model_res, measure = 'precision_2')

# + hidden=true
# Evaluate model on test set
model.evaluate(X_test, y_test)

# + hidden=true
# Calculate predictions
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# + hidden=true
# Check whether there is a clear decision by the network and which industry it refers to
bests = []
for i in range(y_test_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_test_pred[i]) else 0 for j in y_test_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests.append(best)

pd.Series(bests).apply(len).value_counts()

bests_train = []
for i in range(y_train_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests_train.append(best)

pd.Series(bests_train).apply(len).value_counts()

# + hidden=true
# Index of best prediction
best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])

# + hidden=true
# Drop unpredictable cases
best_index_clear = best_index[~np.isnan(best_index)]
y_test_clear = y_test[~np.isnan(best_index)]

best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
y_train_clear = y_train[~np.isnan(best_train_index)]


# + hidden=true
# Check if best prediction is among true technologies
trues = []
for i in range(len(y_test_clear)):
    true = y_test_clear[i, int(best_index_clear[i])]
    trues.append(true)

trues_train = []
for i in range(len(y_train_clear)):
    true = y_train_clear[i, int(best_train_index_clear[i])]
    trues_train.append(true)

# + hidden=true
# Final result
print('{:.0%}'.format(sum(trues)/len(trues)) + ' of top technology prediction on test data is indeed among the top technologies.')
print('{:.0%}'.format(sum(trues_train)/len(trues_train)) + ' of top technology prediction on trainig data is indeed among the top technologies.')

# + [markdown] hidden=true
# Better!

# + [markdown] heading_collapsed=true
# ## 4. Iteration 

# + [markdown] hidden=true
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 2 hidden layers, 128 neurons, multi-label, no drop-out
# - __Loss__: Binary cross entropy

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu")(inputs)
#x = tf.keras.layers.Dropout(0.5)(x)

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu")(x)
# x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Plot the network
#tf.keras.utils.plot_model(model)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold = 0.5), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_test, y_test)
          , batch_size=32
          , epochs=epochs
          , verbose=0)

# + hidden=true
plot_history(model_res, measure = 'precision_3')

# + hidden=true
# Evaluate model on test set
model.evaluate(X_test, y_test)

# + hidden=true
# Calculate predictions
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# + hidden=true
# Check whether there is a clear decision by the network and which industry it refers to
bests = []
for i in range(y_test_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_test_pred[i]) else 0 for j in y_test_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests.append(best)

pd.Series(bests).apply(len).value_counts()

bests_train = []
for i in range(y_train_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests_train.append(best)

pd.Series(bests_train).apply(len).value_counts()

# + hidden=true
# Index of best prediction
best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])

# + hidden=true
# Drop unpredictable cases
best_index_clear = best_index[~np.isnan(best_index)]
y_test_clear = y_test[~np.isnan(best_index)]

best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
y_train_clear = y_train[~np.isnan(best_train_index)]


# + hidden=true
# Check if best prediction is among true technologies
trues = []
for i in range(len(y_test_clear)):
    true = y_test_clear[i, int(best_index_clear[i])]
    trues.append(true)

trues_train = []
for i in range(len(y_train_clear)):
    true = y_train_clear[i, int(best_train_index_clear[i])]
    trues_train.append(true)

# + hidden=true
# Final result
print('{:.0%}'.format(sum(trues)/len(trues)) + ' of top technology prediction on test data is indeed among the top technologies.')
print('{:.0%}'.format(sum(trues_train)/len(trues_train)) + ' of top technology prediction on trainig data is indeed among the top technologies.')

# + [markdown] hidden=true
# Worse!

# + [markdown] heading_collapsed=true
# ## 5. Iteration 

# + [markdown] hidden=true
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 2 hidden layers, 128 neurons, multi-label, drop-out & regularization
# - __Loss__: Binary cross entropy

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1())(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1())(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Plot the network
#tf.keras.utils.plot_model(model)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold = 0.5), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_test, y_test)
          , batch_size=32
          , epochs=epochs
          , verbose=0)

# + hidden=true
plot_history(model_res, measure = 'precision_4')

# + hidden=true
# Evaluate model on test set
model.evaluate(X_test, y_test)

# + hidden=true
# Calculate predictions
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# + hidden=true
# Check whether there is a clear decision by the network and which industry it refers to
bests = []
for i in range(y_test_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_test_pred[i]) else 0 for j in y_test_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests.append(best)

pd.Series(bests).apply(len).value_counts()

bests_train = []
for i in range(y_train_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests_train.append(best)

pd.Series(bests_train).apply(len).value_counts()

# + hidden=true
# Index of best prediction
best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])

# + hidden=true
# Drop unpredictable cases
best_index_clear = best_index[~np.isnan(best_index)]
y_test_clear = y_test[~np.isnan(best_index)]

best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
y_train_clear = y_train[~np.isnan(best_train_index)]


# + hidden=true
# Check if best prediction is among true technologies
trues = []
for i in range(len(y_test_clear)):
    true = y_test_clear[i, int(best_index_clear[i])]
    trues.append(true)

trues_train = []
for i in range(len(y_train_clear)):
    true = y_train_clear[i, int(best_train_index_clear[i])]
    trues_train.append(true)

# + hidden=true
# Final result
print('{:.0%}'.format(sum(trues)/len(trues)) + ' of top technology prediction on test data is indeed among the top technologies.')
print('{:.0%}'.format(sum(trues_train)/len(trues_train)) + ' of top technology prediction on trainig data is indeed among the top technologies.')

# + [markdown] hidden=true
# Worse!
# No precition above 0.5!

# + [markdown] heading_collapsed=true
# ## 6. Iteration 

# + [markdown] hidden=true
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 2 hidden layers, 128 neurons, multi-label, drop-out & less regularization
# - __Loss__: Binary cross entropy

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Plot the network
#tf.keras.utils.plot_model(model)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold = 0.5), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_test, y_test)
          , batch_size=32
          , epochs=epochs
          , verbose=0)

# + hidden=true
plot_history(model_res, measure = 'precision_5')

# + hidden=true
# Evaluate model on test set
model.evaluate(X_test, y_test)

# + hidden=true
# Calculate predictions
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# + hidden=true
# Check whether there is a clear decision by the network and which industry it refers to
bests = []
for i in range(y_test_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_test_pred[i]) else 0 for j in y_test_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests.append(best)

pd.Series(bests).apply(len).value_counts()

bests_train = []
for i in range(y_train_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests_train.append(best)

pd.Series(bests_train).apply(len).value_counts()

# + hidden=true
# Index of best prediction
best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])

# + hidden=true
# Drop unpredictable cases
best_index_clear = best_index[~np.isnan(best_index)]
y_test_clear = y_test[~np.isnan(best_index)]

best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
y_train_clear = y_train[~np.isnan(best_train_index)]


# + hidden=true
# Check if best prediction is among true technologies
trues = []
for i in range(len(y_test_clear)):
    true = y_test_clear[i, int(best_index_clear[i])]
    trues.append(true)

trues_train = []
for i in range(len(y_train_clear)):
    true = y_train_clear[i, int(best_train_index_clear[i])]
    trues_train.append(true)

# + hidden=true
# Final result
print('{:.0%}'.format(sum(trues)/len(trues)) + ' of top technology prediction on test data is indeed among the top technologies.')
print('{:.0%}'.format(sum(trues_train)/len(trues_train)) + ' of top technology prediction on trainig data is indeed among the top technologies.')

# + [markdown] hidden=true
# Better!

# + [markdown] heading_collapsed=true
# ## 7. Iteration 

# + [markdown] hidden=true
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 2 hidden layers, 128 neurons, multi-label, drop-out & regularization l2
# - __Loss__: Binary cross entropy

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l2(l=0.001))(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l2(l=0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Plot the network
#tf.keras.utils.plot_model(model)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold = 0.5), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_test, y_test)
          , batch_size=32
          , epochs=epochs
          , verbose=0)

# + hidden=true
plot_history(model_res, measure = 'precision')

# + hidden=true
# Evaluate model on test set
model.evaluate(X_test, y_test)

# + hidden=true
# Calculate predictions
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# + hidden=true
# Check whether there is a clear decision by the network and which industry it refers to
bests = []
for i in range(y_test_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_test_pred[i]) else 0 for j in y_test_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests.append(best)

pd.Series(bests).apply(len).value_counts()

bests_train = []
for i in range(y_train_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests_train.append(best)

pd.Series(bests_train).apply(len).value_counts()

# + hidden=true
# Index of best prediction
best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])

# + hidden=true
# Drop unpredictable cases
best_index_clear = best_index[~np.isnan(best_index)]
y_test_clear = y_test[~np.isnan(best_index)]

best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
y_train_clear = y_train[~np.isnan(best_train_index)]


# + hidden=true
# Check if best prediction is among true technologies
trues = []
for i in range(len(y_test_clear)):
    true = y_test_clear[i, int(best_index_clear[i])]
    trues.append(true)

trues_train = []
for i in range(len(y_train_clear)):
    true = y_train_clear[i, int(best_train_index_clear[i])]
    trues_train.append(true)

# + hidden=true
# Final result
print('{:.0%}'.format(sum(trues)/len(trues)) + ' of top technology prediction on test data is indeed among the top technologies.')
print('{:.0%}'.format(sum(trues_train)/len(trues_train)) + ' of top technology prediction on trainig data is indeed among the top technologies.')

# + [markdown] hidden=true
# Worse!

# + [markdown] heading_collapsed=true
# ## 8. Iteration 

# + [markdown] hidden=true
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 2 hidden layers, 128 neurons, multi-label, drop-out & L1 regularization, keep best model w.r.t. number of epochs
# - __Loss__: Binary cross entropy

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Plot the network
#tf.keras.utils.plot_model(model)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_dev, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          , callbacks=callbacks_list)

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Evaluate model on test set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_dev, y_dev)

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev)
y_train_pred = model.predict(X_train)


# + hidden=true
def adjusted_metric(y_train, )


# + hidden=true
# Check whether there is a clear decision by the network and which industry it refers to
bests = []
for i in range(y_dev_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_dev_pred[i]) else 0 for j in y_dev_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests.append(best)

pd.Series(bests).apply(len).value_counts()

bests_train = []
for i in range(y_train_pred.shape[0]):
    best = list(np.asarray([1 if j>0.5 and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
    best = np.argwhere(best == np.amax(best)).flatten()
    bests_train.append(best)

pd.Series(bests_train).apply(len).value_counts()

# + hidden=true
# Index of best prediction
best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])

# + hidden=true
# Drop unpredictable cases
best_index_clear = best_index[~np.isnan(best_index)]
y_dev_clear = y_dev[~np.isnan(best_index)]

best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
y_train_clear = y_train[~np.isnan(best_train_index)]


# + hidden=true
# Check if best prediction is among true technologies
trues = []
for i in range(len(y_dev_clear)):
    true = y_dev_clear[i, int(best_index_clear[i])]
    trues.append(true)

trues_train = []
for i in range(len(y_train_clear)):
    true = y_train_clear[i, int(best_train_index_clear[i])]
    trues_train.append(true)

# + hidden=true
# Final result
print('{:.0%}'.format(sum(trues)/len(trues)) + ' of top technology prediction on validation data is indeed among the top technologies.')
print('{:.0%}'.format(sum(trues_train)/len(trues_train)) + ' of top technology prediction on trainig data is indeed among the top technologies.')

# + [markdown] hidden=true
# Better!

# + [markdown] heading_collapsed=true
# ## 9. Iteration 

# + [markdown] hidden=true
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 2 hidden layers, 500 neurons, multi-label, drop-out & L1 regularization, keep best model w.r.t. number of epochs
# - __Loss__: Binary cross entropy
# - __Metric__: user defined F1-Score

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(500, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Vanilla hidden layer:
x = tf.keras.layers.Dense(500, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_dev, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          , callbacks=callbacks_list)

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Evaluate model on validation set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_dev, y_dev)

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev)
y_train_pred = model.predict(X_train)

# + hidden=true
# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, beta=1)

# + [markdown] hidden=true
# Worse!

# + [markdown] heading_collapsed=true
# ## 10. Iteration


# + [markdown] hidden=true
# - __Data__: scraped 50 websites
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 1 hidden layers, 128 neurons, multi-label, drop-out & L1 regularization, keep best model w.r.t. number of epochs
# - __Loss__: Binary cross entropy
# - __Metric__: user defined F1-Score

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights2.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_dev, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          , callbacks=callbacks_list)

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Evaluate model on validation set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_dev, y_dev)

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev)
y_train_pred = model.predict(X_train)

# + hidden=true
# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, threshold=0.5)

# + [markdown] hidden=true
# Better!

# + [markdown] heading_collapsed=true
# ## 11. Iteration


# + [markdown] hidden=true
# - __Data__: scraped 50 websites
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 1 hidden layers, 128 neurons, multi-label, drop-out & L1 regularization, keep best model w.r.t. number of epochs
# - __Loss__: Binary cross entropy
# - __Metric__: user defined F1-Score
# - __Optimizer__ : ADAM
# - __Tuning__: decaying learning rate

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.01/epochs), 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights2.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_dev, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          , callbacks=callbacks_list)

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Evaluate model on validation set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_dev, y_dev)

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev)
y_train_pred = model.predict(X_train)

# + hidden=true
# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, threshold=0.5)

# + [markdown] hidden=true
# Worse!
# -

# ## 12. Iteration: BEST


# - __Data__: scraped 50 websites
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 1 hidden layers, 128 neurons, multi-label, drop-out & L1 regularization, keep best model w.r.t. number of epochs
# - __Loss__: Binary cross entropy
# - __Metric__: user defined F1-Score
# - __Optimizer__ : ADAM
# - __Tuning__: -

# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# +
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)
# -

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights2.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# +
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_dev, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          , callbacks=callbacks_list)
# -

plot_history(model_res, measure = 'f1_score')

# Evaluate model on validation set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_dev, y_dev)

# Calculate predictions
y_dev_pred = model.predict(X_dev)
y_train_pred = model.predict(X_train)

# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, threshold=0.5)

# Better!

# + [markdown] heading_collapsed=true
# ## 13. Iteration


# + [markdown] hidden=true
# - __Data__: scraped 50 websites
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 1 hidden layers, 128 neurons, multi-label, drop-out & L1 regularization on both weights and activation output, keep best model w.r.t. number of epochs
# - __Loss__: Binary cross entropy
# - __Metric__: user defined F1-Score
# - __Optimizer__ : ADAM
# - __Tuning__: -

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, 
                          activation="relu", 
                          activity_regularizer = tf.keras.regularizers.l1(l=0.001),
                          kernel_regularizer = tf.keras.regularizers.l1(l=0.001))(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights2.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_dev, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          , callbacks=callbacks_list)

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Evaluate model on validation set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_dev, y_dev)

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev)
y_train_pred = model.predict(X_train)

# + hidden=true
# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, threshold=0.5)

# + [markdown] hidden=true
# Worse!

# + [markdown] heading_collapsed=true
# ## 14. Iteration


# + [markdown] hidden=true
# - __Data__: scraped 50 websites
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 1 hidden layers, 128 neurons, multi-label, drop-out & L1 regularization on weights only, keep best model w.r.t. number of epochs
# - __Loss__: Binary cross entropy
# - __Metric__: user defined F1-Score
# - __Optimizer__ : ADAM
# - __Tuning__: -

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, 
                          activation="relu", 
                          #activity_regularizer = tf.keras.regularizers.l1(l=0.001),
                          kernel_regularizer = tf.keras.regularizers.l1(l=0.001))(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights2.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_dev, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          , callbacks=callbacks_list)

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Evaluate model on validation set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_dev, y_dev)

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev)
y_train_pred = model.predict(X_train)

# + hidden=true
# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, threshold=0.5)

# + [markdown] hidden=true
# Worse!

# + hidden=true


# + [markdown] heading_collapsed=true
# ## 15. Iteration


# + [markdown] hidden=true
# - __Data__: scraped 50 websites
# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 1 hidden layers, 128 neurons, multi-label, drop-out & L1 regularization on weights only, keep best model w.r.t. number of epochs
# - __Loss__: Binary cross entropy
# - __Metric__: user defined F1-Score
# - __Optimizer__ : ADAM
# - __Tuning__: tune learning rate

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim


# + hidden=true
# Build neural network
def model_builder(hp):
    inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

    # Vanilla hidden layer:
    x = tf.keras.layers.Dense(128, 
                          activation="relu", 
                          activity_regularizer = tf.keras.regularizers.l1(l=0.001))(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Project onto a single unit output layer, and squash it with a sigmoid:
    predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)
    
    # Define final model by specifying input and output layer
    model = tf.keras.Model(inputs, predictions)
    
    # Define learning rate as tunable hyper parameter
    hp_learning_rate = hp.Float(name = 'learning_rate', min_value = 0.0001, max_value = 0.01, sampling = 'log', default = 0.001)
    
    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate, beta_1=0.9, beta_2=0.999), 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
    
    return model


# + hidden=true
# Summary of the model
model.summary()

# + hidden=true
# Define search strategy for tuning hyperparameters
model_tuner = kt.Hyperband(model_builder,
                          objective = 'val_loss',
                          max_epochs = 10,
                          factor = 3,
                          directory=os.path.normpath(r"H:\Keras"),
                          project_name = "Iteration_15")


# + hidden=true
# Define a callback to clear the training outputs at the end of every training step
class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)


# + hidden=true
model_tuner.search(X_train, y_train, epochs = 10, validation_data = (X_dev, y_dev), callbacks = [ClearTrainingOutput()])

# + hidden=true
# Get the optimal hyperparameters
best_hps = model_tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# + hidden=true
# Best hyperparameter to model
model = model_tuner.hypermodel.build(best_hps)

# + hidden=true
# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights2.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_dev, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          , callbacks=callbacks_list)

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Evaluate model on validation set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_dev, y_dev)

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev)
y_train_pred = model.predict(X_train)

# + hidden=true
# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, threshold=0.5)

# + [markdown] hidden=true
# No change!

# + hidden=true


# + hidden=true


# + [markdown] heading_collapsed=true
# ## 11. Iteration 


# + [markdown] hidden=true
# - __Input__: Top Normed technology group counts
# - __Model__: CNN
# - __Loss__: Binary cross entropy
# - __Metric__: user defined F1-Score

# + hidden=true
with open(r"H:\Large_Datasets\Glove\glove.6B.100d.txt", 'r', encoding="utf8") as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
        try:
            print(line)
        except:
            pass
#         line = line.strip().split()
#         curr_word = line[0]
#         words.add(curr_word)
#         word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
#         i = 1
#         words_to_index = {}
#         index_to_words = {}
#         for w in sorted(words):
#             words_to_index[w] = i
#             index_to_words[i] = w
#             i = i + 1
#     return words_to_index, index_to_words, word_to_vec_map

# + hidden=true
# Prepare word embeddings
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(r"H:\Large_Datasets\Glove\glove.6B.50d.txt")

# + hidden=true


# + hidden=true
word = "tanz"
idx = 333
print("the index of", word, "in the vocabulary is", word_to_index[word])
print("the", str(idx) + "th word in the vocabulary is", index_to_word[idx])

# + hidden=true
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# + hidden=true
# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# + hidden=true
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# + hidden=true
# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# + hidden=true
epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_dev, y_dev)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          , callbacks=callbacks_list)

# + hidden=true
plot_history(model_res, measure = 'f1_score')

# + hidden=true
# Evaluate model on validation set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_dev, y_dev)

# + hidden=true
# Calculate predictions
y_dev_pred = model.predict(X_dev)
y_train_pred = model.predict(X_train)

# + hidden=true
# Final results
adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, threshold=0.5)

# + [markdown] hidden=true
# Better!

# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true



# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true



# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true

