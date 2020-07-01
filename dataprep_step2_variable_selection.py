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

df_epovvc = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_epovvc_s1.pkl")

df_epovvc.head(3)


# Function to extract relevant data
def func_extractor(df, cols):
    res = df.loc[:, cols].drop_duplicates()
    return res


df = func_extractor(df_epovvc, cols = ['crefo', 'technology', 'text'])

df.head(3)
