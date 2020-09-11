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

import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
nltk.data.path.append(r'Q:\Meine Bibliotheken\Research\Data\NLTK')

df_epovvc = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_epovvc_s1_d50&cdesc.pkl")

df_epovvc.head(3)

# Distribution ot technology classes
df_epovvc[['crefo', 'appln_id', 'technology', 'text']].drop_duplicates().technology.astype(str).apply(lambda x: x[:2]).value_counts().plot.bar()

# Shape of relevant data
df_epovvc[['crefo', 'appln_id', 'ipc', 'text']].drop_duplicates().shape

# Approach to vectorize technology classes I
np.eye(4)[-1]

# Approach to vectorize technology classes II
np.eye(4)[np.array([i-1 for i in [0,1,3,1,1,1,1]], dtype=int)].sum(axis=0, dtype=int)


# +
# Function clean text data from stopwords
def clean_stopword(text):
    """
    Coerce string to list of tokens seperated by whitespace, 
    use list comprehension to get rid of stopwords, 
    join list of words back to string
    
    Arguments:
    text -- pandas column of dtype string
    
    Returns:
    text -- cleaned string variable without any stopwords
    
    """

    text = [w for w in text.split() if w.lower() not in stopword_list]
    
    text = ' '.join(text)
    return text

# Define stopword lists 
stopwords_en = stopwords.words('english')
stopwords_de = stopwords.words('german')
stopwords_plus = ['de', 'com', 'kannst', 'cookie', 'cookies', 'zb', 'domain', 'browser', 'mm', 'www', 'tel', 'https', 'id', 'impressum']
stopword_list = stopwords_en + stopwords_de + stopwords_plus


# -

# Function to extract relevant data
def func_extractor(df, Y, X, top_n = 3, drop_stop = True):
    """
    Prepare data for model training. 
    Choosing features X and target variable Y. 
    Select variables accordingly and 
    conduct further preprocessing steps such as stop word deletion.
    
    Arguments:
    df -- pandas DataFrame
    Y -- target variable
    X -- feature variable(s)
    top_n -- number of most important technologies to consider per crefo
    drop_stop -- drop stop words (not working reliably)
    
    Returns:
    df_train -- DataFrame including the following forms of target vectors
        y_count -- number of occurences of target
        y_bool -- indicator of occurences of target
        y_norm -- normed number of occurences of target
        y_norm_top -- normed number of occurences of top_n targets
        y_mhot_top -- multiple-hot representation of top_n target
        y_ohot -- multiple-hot representation of most important technology (multiple because two ore more technologies can draw)
        
        text -- webdata
        clean_text -- webdata without stopwords
    
    """
    
    # Select variables and drop duplicated rows
    res = df.loc[:, ['crefo', 'appln_id', Y, X]].drop_duplicates()
    dim1 = len(res)
    
    # Drop rows with missing values in features
    res = res.loc[res[X].notnull(), :]
    dim2 = len(res)
    print(str(dim1 - dim2) + ' samples dropped due to missings in X.')
    
    # Drop rows with missing values in target
    res = res.loc[res[Y].notnull(), :]
    dim3 = len(res)
    print(str(dim2 - dim3) + ' further samples dropped due to missings in Y.')
    
    # Calculate number of distinct classes
    n_classes = len(res[Y].unique())
    print('There are ' + str(n_classes) + ' distinct classes.')
    
    # Prepare target as vector of length = n_classes on crefo-level
    y = res.groupby('crefo')[Y].apply(lambda x: np.eye(n_classes)[np.array([i-1 for i in x], dtype=int)])

    # Calculate number of occurences of target
    y_count = y.apply(lambda x: sum(x),)
    # Calculate indicator of occurences of target
    y_bool = y_count.apply(lambda x: [1 if i>0 else 0 for i in x])
    # Calculate normed number of occurences of target
    y_norm = y_count.apply(lambda x: [i/max(x) for i in x])
    # Calculate normed number of occurences of top_n targets
    y_norm_top = y_count.apply(lambda x: [i/max(sorted(x, reverse = True)[:top_n]) if i in sorted(x, reverse = True)[:3] else 0 for i in x])
    # Calculate multiple-hot representation of top_n targets
    y_mhot_top = y_norm_top.apply(lambda x: [1 if i>0 else 0 for i in x])
    # Calculate multiple-hot representation of most important techology(ies)
    y_mhot = y_count.apply(lambda x: [1 if i == max(x) else 0 for i in x])
    # Create DataFrame including all target variations
    ys = pd.concat([y_bool, y_count, y_norm, y_norm_top, y_mhot_top, y_mhot], axis=1, keys=['y_true_bool', 'y_true_count', 'y_true_norm', 'y_true_up', 'y_true_up_bool', 'y_true_top_bool'])
    
    # Drop stopwords
    if drop_stop:
        res.loc[:,'clean_text'] = res.text.apply(lambda x: clean_stopword(x))
        df_train = ys.merge(res[['crefo', 'clean_text', X]].drop_duplicates(ignore_index=True), left_index=True, right_on='crefo', how='left')
        
    else:
        df_train = ys.merge(res[['crefo', X]].drop_duplicates(ignore_index=True), left_index=True, right_on='crefo', how='left')
    
    # Set crefo as index
    df_train.set_index('crefo', inplace=True)
        
    return df_train


df_epovvc.head(3)

df_train = func_extractor(df_epovvc, Y = 'technology', X = 'cdesc', drop_stop = False)

df_train.shape

df_train.loc[:,'clean_text'] = df_train.text.apply(lambda x: clean_stopword(x))

df_train.head()

# Reduce to y_true_up_bool
df_train = df_train.loc[:,['y_true_up_bool', 'clean_text']]

# Write data
with open(r'Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_train_cdesc.pkl', 'wb') as f:
    pickle.dump(obj=df_train, file=f)
