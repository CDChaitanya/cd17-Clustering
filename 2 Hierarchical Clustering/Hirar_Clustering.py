# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[: , [3,4]].values

#USING DENDOGRAM TO FIND OPTIMAL NUMBER OF CLUSTERS
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X , method ='ward'))
plt.title("DENDOGRAM")
plt.xlabel("COUSTOMERS")
plt.ylabel("EUCLIDIAN DISTANCE")
plt.show()
#FROM DENDO WE CAN SAY THAT THE NO. OF CLUSTER ARE 5

#FITTING HIRARCHICAL CLUSTERING TO MALL DATASET
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5 , affinity="euclidean" , linkage='ward')
y_hc = hc.fit_predict(X)

#Visualizing the cluster 
plt.scatter(X[y_hc==0 , 0], X[y_hc==0 , 1], s=100 ,c="red"     ,label="CAREFUL  ")
plt.scatter(X[y_hc==1 , 0], X[y_hc==1 , 1], s=100 ,c="blue"    ,label="STANDARD ")
plt.scatter(X[y_hc==2 , 0], X[y_hc==2 , 1], s=100 ,c="green"   ,label="TARGET   ")
plt.scatter(X[y_hc==3 , 0], X[y_hc==3 , 1], s=100 ,c="cyan"    ,label="CARELESS ")
plt.scatter(X[y_hc==4 , 0], X[y_hc==4 , 1], s=100 ,c="magenta" ,label="SENSIBLE ")
plt.title("CLUSTER OF CLIENTS")
plt.xlabel("ANUAL INCOME [K$]")
plt.ylabel("SPENDING SCORE ")
plt.legend()
plt.show() 