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

# # EPO - MUP Data 

# Read epo to mup data
df_epovvc = pd.read_csv(r"V:\midp\match\epo2019\epovvc2019.txt"
                        ,sep="\t"
                        ,encoding="utf-8")

df_epovvc.shape

df_epovvc.head()

# One crefo may have many patents (different appln_id/person_id)
df_epovvc.loc[df_epovvc.crefo==2150355978,:]

# There are only 1s for bestcrefo. No need to drop rows
df_epovvc.bestcrefo.value_counts(dropna=False)

# Trainings data comprises 36868 firms with known patent match
len(pd.unique(df_epovvc.crefo))

df_tech = pd.read_csv(r"V:\midp\match\epo2019\tech.txt"
                     ,sep="\t"
                     ,encoding="utf-8")

# One application ID may be assigned to several technology classes
df_tech.head()

df_tech.shape

# There are 35 distinct technology classes
len(pd.unique(df_tech.technology))

df_ipc = pd.read_csv(r"V:\midp\match\epo2019\ipc.txt"
                     ,sep="\t"
                     ,encoding="utf-8")

df_ipc.head()

df_ipc.shape

# There are 59915 different ipc classes
len(pd.unique(df_ipc.ipc))

# There are 123 different ipc classes (3-digit level)
len(pd.unique(df_ipc.ipc.apply(lambda x: x[0:3])))

# # Merge Data 

# Merge both datasets
df = df_epovvc.merge(df_tech, how = "left").astype({"technology": "object"}).merge(df_ipc, how = "left")

df.head()

df.shape

# Write data
with open(r'Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\df_epovvc_s0.pkl', 'wb') as f:
    pickle.dump(obj=df, file=f)

# # Crefo & URL for Scraping 

df_url = pd.read_csv(r"I:\!Projekte\BMBF_TOBI_131308\01_Arbeitspakete\01_Webscraper\Webscraper\URLs\2019_lebende\MUP_2019_URLs_raw.txt"
                    ,sep="\t"
                    ,encoding="latin-1"
                    ,header=None
                    ,names=['crefo', 'url'] )

# Strip whitespaces in url column
df_url["url"] = df_url.url.apply(lambda x: x.strip())

# Merge with crefos
df_crefo = pd.DataFrame(pd.unique(df.crefo), columns={"crefo"}).merge(df_url, how = "left")

# Filter only crefos with url
df_crefo = df_crefo.loc[df_crefo.url.notnull(),:]

# Write data
df_crefo.to_csv(r"Q:\Meine Bibliotheken\Research\Projects\Ausgründungen\02_Data\02_Webdata2IPC\urls.txt"
               ,sep = "\t"
               ,encoding="utf-8"
               ,index=False)
