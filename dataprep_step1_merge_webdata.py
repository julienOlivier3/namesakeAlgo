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
# Define which depth of scraping you want to take
depth = 50 # 10 or 50
df_web = []

for i in range(1, 8):
    
    df_web.append(pd.read_csv(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\webdata\depth" + str(depth) + r"\ARGUS_chunk_p" + str(i) + ".csv"
                 ,sep="\t"
                 ,encoding="utf-8"))
    
# -

# Concatenate list elements
df_web = pd.concat(df_web)

# Drop rows with NaN in text column
df_web = df_web.loc[df_web.text.notnull(),:]

df_web.ID = df_web.ID.astype(str)

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

df_web.head()

# # Company Descriptions 

df_epovvc = pd.read_pickle(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_epovvc_s0.pkl")

df_epovvc.crefo = df_epovvc.crefo.astype(str)

df_epovvc.head(3)

# +
df_cdescs = []
crefos = df_epovvc.crefo.astype(str).unique()
#i = 0

for chunk in pd.read_csv(r"I:\JDO\texte_taetigkeit_alle2.lst"
            ,sep="\t"
            ,encoding="latin-1"
            ,dtype={'crefo      ':object,
                   ' gesch_gegen':str
                
            }
            #,nrows=100000          
            ,chunksize=100000
                      ):

    # Rename crefo column name and description column name
    chunk.rename(columns={'crefo      ':'crefo',
                         ' gesch_gegen':'cdesc'}, inplace=True)

    # Clean crefo column
    chunk['crefo']=chunk['crefo'].apply(lambda x: str(x).strip()).astype(str)

    # Filter relevant crefos
    df_cdesc = chunk.loc[chunk.crefo.isin(crefos),:]
    df_cdescs.append(df_cdesc)
    
    # Controls
    #print(i)
    print(pd.concat(df_cdescs).shape)
    #i=+1
    
# -

# Drop last row (contains no info)
df_cdesc= pd.concat(df_cdescs)

# # Merge Data 

# Merge web data
df = df_epovvc.merge(df_web, how = "left")

df.head(3)

# Merge company descriptions
df = df.merge(df_cdesc, how = "left", left_on='crefo', right_on='crefo')

# Fraction of samples without company description
df.cdesc.isnull().sum()/len(df)

# Fraction of samples without webdata
df.text.isnull().sum()/len(df)

# Number of unique firms with webdata/company descriptions
df.loc[df.text.notnull(),:].crefo.unique().shape, df.loc[df.cdesc.notnull(),:].crefo.unique().shape

df.sample(3)

# Write data
with open(r'Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_epovvc_s1_d50&cdesc.pkl', 'wb') as f:
    pickle.dump(obj=df, file=f)
