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
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

df = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_train.pkl")

df.head(1)

# Train test split
train, test = train_test_split(df, test_size = 0.1, random_state=333)

len(df.loc[:,"text"].values)

len(np.stack(list(df.loc[:,"technology"].values), axis=0))
