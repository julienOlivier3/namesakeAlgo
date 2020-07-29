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

import tensorflow as tf
import tensorflow_addons as tfa
import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# -

df = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\dict_train_dev_test_s0.pkl")

X_train = df['X_train']
X_dev = df['X_dev']
X_test = df['X_test']
y_train = df['y_train']
y_dev = df['y_dev']
y_test = df['y_test']

# + [markdown] heading_collapsed=true
# # Functions

# + hidden=true
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


# + hidden=true
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


# -

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
# -

# ## 8. Iteration 

# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 2 hidden layers, 500 neurons, multi-label, drop-out & L1 regularization, keep best model w.r.t. number of epochs
# - __Loss__: Binary cross entropy

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

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# +
# Plot the network
#tf.keras.utils.plot_model(model)
# -

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights.hdf5"
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

# Evaluate model on test set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_dev, y_dev)

# Calculate predictions
y_dev_pred = model.predict(X_dev)
y_train_pred = model.predict(X_train)


def adjusted_metric(y_train, )


# +
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
# -

# Index of best prediction
best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])

# +
# Drop unpredictable cases
best_index_clear = best_index[~np.isnan(best_index)]
y_dev_clear = y_dev[~np.isnan(best_index)]

best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
y_train_clear = y_train[~np.isnan(best_train_index)]


# +
# Check if best prediction is among true technologies
trues = []
for i in range(len(y_dev_clear)):
    true = y_dev_clear[i, int(best_index_clear[i])]
    trues.append(true)

trues_train = []
for i in range(len(y_train_clear)):
    true = y_train_clear[i, int(best_train_index_clear[i])]
    trues_train.append(true)
# -

# Final result
print('{:.0%}'.format(sum(trues)/len(trues)) + ' of top technology prediction on validation data is indeed among the top technologies.')
print('{:.0%}'.format(sum(trues_train)/len(trues_train)) + ' of top technology prediction on trainig data is indeed among the top technologies.')

# Better!

# ## 9. Iteration 

# - __Input__: Top Normed technology group counts
# - __Model__: ANN, 2 hidden layers, 500 neurons, multi-label, drop-out & L1 regularization, keep best model w.r.t. number of epochs
# - __Loss__: Binary cross entropy
# - __Metric__: user defined F1-Score

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

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# +
# Plot the network
#tf.keras.utils.plot_model(model)
# -

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer="adam", 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights.hdf5"
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

# Evaluate model on test set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_dev, y_dev)

# Calculate predictions
y_dev_pred = model.predict(X_dev)
y_train_pred = model.predict(X_train)


def adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, beta=1):
    # Check whether there is a clear decision by the network and which industry it refers to
    bests = []
    for i in range(y_dev_pred.shape[0]):
        best = list(np.asarray([1 if j>0.5 and j==max(y_dev_pred[i]) else 0 for j in y_dev_pred[i]]))
        best = np.argwhere(best == np.amax(best)).flatten()
        bests.append(best)

    bests_train = []
    for i in range(y_train_pred.shape[0]):
        best = list(np.asarray([1 if j>0.5 and j==max(y_train_pred[i]) else 0 for j in y_train_pred[i]]))
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
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F_beta = (1 + beta**2)*(Precision*Recall)/((beta**2 * Precision) + Recall)
    
    return Precision, Recall, F_beta

adjusted_metric(y_train, y_dev, y_train_pred, y_dev_pred, beta=1)

# +
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
# -

# Index of best prediction
best_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests])
best_train_index = np.asarray([i[0] if len(i)==1 else np.nan for i in bests_train])

# +
# Drop unpredictable cases
best_index_clear = best_index[~np.isnan(best_index)]
y_dev_clear = y_dev[~np.isnan(best_index)]

best_train_index_clear = best_train_index[~np.isnan(best_train_index)]
y_train_clear = y_train[~np.isnan(best_train_index)]


# +
# Check if best prediction is among true technologies
trues = []
for i in range(len(y_dev_clear)):
    true = y_dev_clear[i, int(best_index_clear[i])]
    trues.append(true)

trues_train = []
for i in range(len(y_train_clear)):
    true = y_train_clear[i, int(best_train_index_clear[i])]
    trues_train.append(true)
# -

# Final result
print('{:.0%}'.format(sum(trues)/len(trues)) + ' of top technology prediction on validation data is indeed among the top technologies.')
print('{:.0%}'.format(sum(trues_train)/len(trues_train)) + ' of top technology prediction on trainig data is indeed among the top technologies.')

# Better!

np.isnan(best_index).sum()/len(best_index)
































np.asarray([1 if j>0.5 else 0 for j in y_test_pred[0]]), y_test[0]

X

# +
y_test_pred = model.predict(X_test)
accs = []
for i in range(y_test.shape[0]):
    u = y_test[i]
    v = np.asarray([1 if j>0.5 else 0 for j in y_test_pred[i]])
    acc = v.sum()
    accs.append(acc)
    
np.mean(accs), np.median(accs)

# +
# Understand accuracy calculation
y_test_pred = model.predict(X_test)
accs = []
for i in range(y_test.shape[0]):
    u = y_test[i]
    v = np.asarray([1 if j>0.5 else 0 for j in y_test_pred[i]])
    acc = np.equal(u, v)
    accs.append(acc)


# Calculate goodness of fit
np.mean(accs)

# +
# Understand recall calculation
u = pd.Series(y_test.reshape(35*len(y_test)))
v = pd.Series(np.asarray([1 if j>0.5 else 0 for j in y_test_pred.reshape(35*len(y_test))]))
col = pd.concat([u,v], axis=1)

# Calculate Recall and Precision
pos = col.loc[col[1]==1,:]
neg = col.loc[col[1]==0,:]
TP = np.equal(pos[0], pos[1]).sum()
TN = np.equal(neg[0], neg[1]).sum()
FP = np.not_equal(pos[0], pos[1]).sum()
FN = np.not_equal(neg[0], neg[1]).sum()

# Recall, Precision
TP/(TP+FN), TP/(TP+FP)
# -

np.arange(12.)*np.pi/6, np.degrees(np.arange(12.)*np.pi/6) 

np.set_printoptions(precision=3, suppress=True)
u,v

v2 = np.random.uniform(size=len(v))
c = np.arccos(np.dot(u,v2)/(np.linalg.norm(u)*np.linalg.norm(v2)))
np.degrees(c)

# +
# Calculate angel between vectors
y_test_pred = model.predict(X_test.toarray().astype(np.float32))
angles = []
for i in range(y_test.shape[0]):
    u = y_test[i]
    v = y_test_pred[i]
    c = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
    angle = np.degrees(np.arccos(c))
    angles.append(angle)


# Calculate goodness of fit
np.mean(angles), sum([1 for i in angles if i < 90])/len(angles)
# -

angles

# Analyze bad classifications


df_train.columns


def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed


# +
# Train test split
df_train, df_test = train_test_split(df, test_size = 0.1, random_state=333)

X_train = df_train.text.to_numpy()
y_train = df_train.y_true_norm.values

X_test = df_test.text.to_numpy()
y_test = df_test.y_true_norm.values

# +
# Convert target as clean numpy array
n = y_train.shape[0]
a = []
for i in range(n):
    a.append([j for j in y_train[i]])
y_train = np.array(a)

# Convert target as clean numpy array
n = y_test.shape[0]
a = []
for i in range(n):
    a.append([j for j in y_test[i]])
y_test = np.array(a)


# +
# Standardize text
def text_standardization(input_data):
    '''
    lowercase, delete html tags, delte whitespaces, delete numbers, delete punctuation
    
    '''
    
    # lowercasing
    clean_text = tf.strings.lower(input_data)
    # delete html tags
    clean_text = tf.strings.regex_replace(clean_text, r"\[->.*?<-\]", "")
    # delete leading and trailing whitespaces
    clean_text = tf.strings.strip(clean_text)
    # delete numbers not part of a word
    clean_text = tf.strings.regex_replace(clean_text, r"\b\d+\b", "")
    # delete punctuation
    clean_text = tf.strings.regex_replace(clean_text, "[%s]" % re.escape(string.punctuation), "")
    return clean_text

#df.loc[:,'clean_text'] = df.clean_text.apply(lambda x: text_standardization(x))


# +
# Model constants.
max_features = 2000
embedding_dim = 128
sequence_length = 500


# Now that we have our custom standardization, we can instantiate our text
# vectorization layer. We are using this layer to normalize, split, and map
# strings to integers, so we set our 'output_mode' to 'int'.
# Note that we're using the default split function,
# and the custom standardization defined above.
# We also set an explicit maximum sequence length, since the CNNs later in our
# model won't support ragged sequences.
vectorize_layer = TextVectorization(
    standardize=text_standardization,
    max_tokens=max_features,
    output_mode="tf-idf",
    #output_sequence_length=sequence_length,
)

# Now that the vocab layer has been created, call `adapt` on a text-only
# dataset to create the vocabulary. You don't have to batch, but for very large
# datasets this means you're not keeping spare copies of the dataset in memory.

# Let's make a text-only dataset (no labels):
#text_ds = raw_train_ds.map(lambda x, y: x)
# Let's call `adapt`:
vectorize_layer.adapt(X_train)
# -

# Take a look at vocabulary
vectorize_layer.get_vocabulary()[:10]

text_vector = vectorize_layer.get_weights()
text_dict = {'Word': text_vector[0][0:len(text_vector[0])], 
             'Vocabulary_index': text_vector[1][0:len(text_vector[1])], 
             'TF-IDF': text_vector[2][1:len(text_vector[2])]}
pd.DataFrame.from_dict(text_dict)

# Specify input and output dimensions
inputs_dim = max_features
outputs_dim = len(y_train[0])
inputs_dim, outputs_dim

# +
# A integer input for vocab indices.
text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')

x = vectorize_layer(text_input)

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
#x = tf.keras.layers.Embedding(max_features, embedding_dim)(x)
#x = tf.keras.layers.Dropout(0.5)(x)

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(text_input, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# +
epochs = 50

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_test, y_test)
          , batch_size=32
          , epochs=epochs)


# +
# Define a text pre-processing function
def clean_text(text):
    '''delete html tags, delete whitespaces, delete trailing whitespaces'''
    text = re.sub(r'\[->.*?<-\]', ' ', text)
    text = text.strip()
    text = re.sub(r' +', ' ', text)
    return text


# In[6]:


# Define function cleaning digits
def clean_digit(text):
    '''Get rid of digits which are not part of a word'''
    text = re.sub(r'\b\d+\b', '', text)
    return text


# In[7]:


# Define function dropping English and German stopwords

# Extend data path list in nltk by the path where the corpora are located
nltk.data.path.append(r'Q:\Meine Bibliotheken\Research\Data\NLTK')
nltk.data.path

# Define lists 
stopwords_en = stopwords.words('english')
stopwords_de = stopwords.words('german')
stopwords_plus = ['de', 'com', 'kannst', 'cookie', 'cookies', 'zb', 'domain', 'browser', 'mm', 'www', 'tel', 'https', 'id', 
                 'impressumimpressum', 'impressum', 'copyright', 'datenschutzerklärungdatenschutzerklärung', 'agbagb',
                 'kontaktkontakt', 'datenschutzdatenschutz']

# Function
def clean_stopword(text):
    '''Coerce string to list of tokens, use list comprehension to get rid of stopwords, join list of words back to string'''
    text = nltk.word_tokenize(text)

    text = [w for w in text if w.lower() not in stopwords_en]
    text = [w for w in text if w.lower() not in stopwords_de]
    text = [w for w in text if w.lower() not in stopwords_plus]
    
    text = ' '.join(text)
    return text


# -

t = Tokenizer(num_words=10000)
t.fit_on_texts(df.text)

X = t.texts_to_matrix(df.text, mode = "tfidf")




# Train test split
train, test = train_test_split(df, test_size = 0.1, random_state=333)

train.head()

len(df.loc[:,"text"].values)

len(np.stack(list(df.loc[:,"technology"].values), axis=0))


