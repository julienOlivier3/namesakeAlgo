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
from keras.preprocessing.text import text_to_word_sequence

df_epovvc = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_epovvc_s1.pkl")

df_epovvc.head(3)


# Function to extract relevant data
def func_extractor(df, cols, Y, X):
    """Prepare data for model training. Choosing features X and target variable Y. Select variables accordingly and 
    conduct further preprocessing steps."""
    
    # Select variables and drop duplicated rows
    res = df.loc[:, cols].drop_duplicates()
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
    
    # Prepare target as multiple-hot vector of length = n_classes
    y = res.groupby('crefo')[Y].apply(lambda x: np.eye(n_classes)[np.array([i-1 for i in x], dtype=int)].sum(axis=0, dtype=int))
    
    
    
    return res, y

df, y = func_extractor(df_epovvc, cols = ['crefo', 'technology', 'text'], Y = 'technology', X = 'text')

y

df.groupby('crefo')['technology'].apply(lambda x: np.eye(35)[np.array([i-1 for i in x], dtype=int)].sum(axis=0, dtype=int))
