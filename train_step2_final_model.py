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
import re
import string
import os
import random
import matplotlib.pyplot as plt
import pickle
import time
import winsound


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
import tensorflow_addons as tfa
import keras
# -

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


# -

# Function measuring execution time
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Done in:', '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))


# Standardize text
class TextStandardizer:
    
    def text_standardization(self, input_data):
        '''
        lowercase, delete html tags, delte whitespaces, delete numbers, delete punctuation
    
        '''
    
        self.input_data = input_data
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
                 + 'Prediction made (prediction above ' + '{:.2f}'.format(threshold) + ') for: ' + '{:.2%}'.format(1-no_pred) + '\n' 
                 + '\n' +
                 'Training set precision: ' + '{:.2%}'.format(Precision_train) + '\n' 
                 + 'Prediction made (prediction above ' + '{:.2f}'.format(threshold) + ') for:  ' + '{:.2%}'.format(1-no_pred_train))

# + [markdown] heading_collapsed=true
# # Train Test Split

# + hidden=true
df = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_train_d50.pkl")

# + hidden=true
df.head(3)

# + hidden=true
# Drop rows with short text data
min_words = 100
row_i = df.clean_text.apply(lambda x: len(x.split()))
df = df.loc[(row_i >= min_words).values,:]

# + hidden=true
df.shape

# + hidden=true
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

# # Training

# + [markdown] heading_collapsed=true
# ## Vectorization 

# + hidden=true
# Define vectorizer
vectorizer = TfidfVectorizer(
                  encoding = 'utf-8'
                , preprocessor = text_standardization
                , max_df = 0.95
                , min_df = 3
                , max_features = 20000
                            )

# + hidden=true
# Train vectorizer
vectorizer.fit(X_train_text)

# + hidden=true
# Save vectorizer
with open(r'Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\05_Model\vectorizer.pkl', 'wb') as f:
    pickle.dump(obj=vectorizer, file=f)

# + hidden=true
# Transform text data according to trained vectorizer
X_train = vectorizer.transform(X_train_text).toarray().astype(np.float32)
X_test = vectorizer.transform(X_test_text).toarray().astype(np.float32)
# -

# ## Neural Network

# +
# Specify input and output dimensions
inputs_dim = X_train.shape[1]
outputs_dim = len(y_train[0])
print('The number of input features is: ' + str(inputs_dim))
print('The number of output nodes is: ' + str(outputs_dim))

# Build neural network
inputs = tf.keras.Input(shape=(inputs_dim,), dtype=tf.float32, name='text')

# Vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu", activity_regularizer = tf.keras.regularizers.l1(l=0.001))(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(outputs_dim, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), 
              #metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
              metrics=[tfa.metrics.F1Score(num_classes=outputs_dim, threshold=0.5, average='macro'), tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), "binary_accuracy"]
             )
model.summary()

# +
# Create checkpoint file which contains best model
filepath=r"Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\02_Data\02_Webdata2IPC\best_weights2.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

epochs = 20

# Fit the model using the train and test datasets.
model_res = model.fit(X_train
          , y_train
          , validation_data=(X_test, y_test)
          , batch_size=32
          , epochs=epochs
          , verbose=0
          , callbacks=callbacks_list)

# -

# Plot epochs
plot_history(model_res, measure = 'f1_score')

# +
# Evaluate model on validation set
# Load weights of best model
model.load_weights(filepath)
model_res.model.evaluate(X_test, y_test)

# Calculate predictions
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
# -

# Save model
#tf.keras.models.save_model(model, r'Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\05_Model')
model.save(r'Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\05_Model\NN.h5')

y_test[2],y_test_pred[2],X_test_text[2]

# Final results
adjusted_metric(y_train, y_test, y_train_pred, y_test_pred, threshold=1/3)

# + [markdown] heading_collapsed=true
# # Model Object 

# + hidden=true
df_temp = pd.DataFrame(data=X_test_text, columns=['text'])
df_temp['crefo'] = range(0,len(X_test_text))
df_temp.set_index('crefo')
df_temp.head(3)


# + hidden=true
def model_iterator(y_pred, prob_threshold, crefo):
    d = {}
    
    for i in range(y_pred.shape[0]):
        best = [1 if j>prob_threshold and j==max(y_pred[i]) else 0 for j in y_pred[i]]
        if any(best):
            best = int(np.argwhere(best).flatten())
            d[crefo[i]] = best
        else:
            pass
    return d 


# + hidden=true
def model_web2tech(df, prob_threshold):
    # VARIABLE DEFINITION
    crefo = df.index
    X = df.text
    
    # TEXT VECTORIZATION
    vectorizer = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\05_Model\vectorizer.pkl")
    X_vec = vectorizer.transform(X).toarray().astype(np.float32)
    
    # NN PREDICTION
    NN = tf.keras.models.load_model(r"Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\05_Model\NN.h5")
    y_pred = NN.predict(X_vec)
    
    # TECHNOLOGY MAPPING
    d = model_iterator(y_pred, prob_threshold, crefo)
    
    return d


# + hidden=true
y_pred_temp = model_web2tech(df_temp, prob_threshold=1/3)

# + [markdown] heading_collapsed=true
# # Loop Webpages 

# + hidden=true
# Loop over all MUP observations in chunks

# Define source folder, file & filetype
folder = r'I:\!Projekte\BMBF_TOBI_131308\01_Arbeitspakete\01_Webscraper\Webscraper\Scrapeyard\MUP Crawl -April 2019 (limit 50, prefer short, no language)\chunks'
file_name = r"\output_p"
file_type = ".csv"

# Define empty list for saving reduced results
d = {}
ds = {}

# Measure start of execution time
t0 = time.time()


# Start of loop
for o in range(81, 117):
    path = folder + file_name + str(o) + file_type
    try:
        df_chunk = pd.read_csv(path
                               , sep=r'\t'
                               , encoding='utf-8'
                               #, nrows=1000
                              )
            
               
        # DROP MISSING TEXTS
        df_chunk = df_chunk.loc[df_chunk['text'].notnull(), ['ID', 'text']].rename(columns = {'ID': 'crefo'})
            
            
        # AGGREGATE BY crefo
        df = df_chunk.groupby('crefo').aggregate(
                dict(
                    text = lambda x: ''.join(x)
                    )
                                            )
        
        # EXTRACT TEXT FRAGMENTS
        d = model_web2tech(df, prob_threshold=1/3)
            
        # APPEND DICTIONARIES
        ds = {**ds, **d}
        
        print(o)
        
    except:
        pass


# Print execution time
timer(t0, time.time())

# Play sound once finished
winsound.PlaySound(r'C:\Users\jdo\Documents\R\R-3.6.2\library\beepr\sounds\smb_coin.wav', winsound.SND_FILENAME)

# + hidden=true
df_tech = pd.DataFrame.from_dict(ds, orient='index', columns=['predicted_tech_class'])

# + hidden=true
df_tech.head()

# + hidden=true
# Write data
with open(r'Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\02_Data\02_Webdata2IPC\01_Predictions\df_tech2.pkl', 'wb') as f:
    pickle.dump(obj=df_tech, file=f)

# + [markdown] heading_collapsed=true hidden=true
# ## Append Results 

# + hidden=true
df_tech1 = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\02_Data\02_Webdata2IPC\01_Predictions\df_tech1.pkl")
df_tech2 = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\02_Projects\Ausgründungen\02_Data\02_Webdata2IPC\01_Predictions\df_tech2.pkl")

# + hidden=true
df_tech1.index.names=['crefo']
df_tech2.index.names=['crefo']

# + hidden=true
df_tech = pd.concat([df_tech1, df_tech2])

# + hidden=true
df_tech.predicted_tech_class.value_counts().sort_index().plot(kind='bar')

# + hidden=true
df_tech.to_csv(r'B:\02_Intermediate\tech_list.txt', sep = '\t', encoding='utf-8')
