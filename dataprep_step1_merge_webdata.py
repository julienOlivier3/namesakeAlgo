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
import math

# # Webdata 

# +
df_web = []

for i in range(1, 8):
    
    df_web.append(pd.read_csv(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\webdata\ARGUS_chunk_p" + str(i) + ".csv"
                 ,sep="\t"
                 ,encoding="utf-8"))
    
# -

df_web = pd.concat(df_web)

# Drop rows with NaN in text column
df_web = df_web.loc[df_web.text.notnull(),:]

df_web.head(3)

# Aggregate webdata by crefo
df_web = df_web.groupby('ID').agg(
    dict(
        text = lambda x: ' '.join(x),
        links = lambda x: list(x)
        )
)

# Rename ID to crefo and define as column
df_web = df_web.reset_index().rename(columns={"ID": "crefo"})

df_web

# # Merge Data 

df_epovvc = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_epovvc_s0.pkl")

df_epovvc.head(3)

df = df_epovvc.merge(df_web, how = "left")

# Write data
with open(r'Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_epovvc_s1.pkl', 'wb') as f:
    pickle.dump(obj=df, file=f)
