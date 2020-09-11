Based on a training data set of firm - patent matches, this repository aims at developing a neural network which maps patent technoloy classes to text data found on firm websites.

The idea is that product descriptions found on company websites allow to assign companies with websites a technology class based on (product) patents in the training set.

The ultimate goal is to develop an algorithm which allows to disambigaute researcher careers by relating patent and publication documents to university spin-offs on the indivdual name level.
For this purpose it requires mutual traits between patents, publications and companies which allow to disambiguate these entities belonging to one and the same person. 
The issue of linking these entities only by means of the names of publication authors, patent inventors and company founders is that entities of namesakes will be matchted. 
Therefore, it requires further linking features (traits) that allow to disambiguat the above entities within name spaces.
