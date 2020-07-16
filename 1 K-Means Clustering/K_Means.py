# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[: , [3,4]].values

#USING ELBOW METHOD TO FIND OPTIMAL NUMBER OF CLUSTERS
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i , init="k-means++" , max_iter= 300 
                    , n_init= 10 , random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11) , wcss)
plt.title("THE ELBOW METHOD")
plt.xlabel("NO. OF CLUSTER")
plt.ylabel("WCSS")
plt.show()     
#FROM GRAPH WE CAN SAY THAT THE NO. OF CLUSTER ARE 5

#APPLYING KMEANS TO MALL DATASET
kmeans = KMeans(n_clusters=5 , init="k-means++" , n_init=10 
                ,max_iter =300, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Visualizing the cluster 
plt.scatter(X[y_kmeans==0 , 0], X[y_kmeans==0 , 1], s=100 ,c="red" ,    label="CAREFUL  ")
plt.scatter(X[y_kmeans==1 , 0], X[y_kmeans==1 , 1], s=100 ,c="blue" ,   label="STANDARD ")
plt.scatter(X[y_kmeans==2 , 0], X[y_kmeans==2 , 1], s=100 ,c="green",   label="TARGET   ")
plt.scatter(X[y_kmeans==3 , 0], X[y_kmeans==3 , 1], s=100 ,c="cyan" ,   label="CARELESS ")
plt.scatter(X[y_kmeans==4 , 0], X[y_kmeans==4 , 1], s=100 ,c="magenta" ,label="SENSIBLE ")
plt.scatter(kmeans.cluster_centers_[: , 0] , kmeans.cluster_centers_[:,1] , s=300 , c="yellow" , label="CENTER")
plt.title("CLUSTER OF CLIENTS")
plt.xlabel("ANUAL INCOME [K$]")
plt.ylabel("SPENDING SCORE ")
plt.legend()
plt.show() 