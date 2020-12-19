import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def init_clustr(K,data):
    clust_dat=pd.DataFrame([])
    randlst = list()
    for i in range(0,K):
        random.seed(i*101)
        t = random.randint(0,199)
        randlst.append(t)
    clust_dat = data.iloc[randlst, :]
    clust_dat = clust_dat.reset_index(drop = True)
    return clust_dat

def clustr_assgn(data,old_clust,clust_dat,i,attrbs,flg):
    distance = 1e9
    ln = len(clust_dat)
    j=0
    while j<ln:
        new_dist = 0
        for att in attrbs:
            new_dist = new_dist + (data.loc[i, att] - clust_dat.loc[j, att])**2
        new_dist = np.sqrt(new_dist)
        if new_dist < distance:
            distance = new_dist
            data.loc[i, 'Clust'] = j
        j+=1
    if (data.loc[i, 'Clust'] != old_clust[i]):
        flg+=1
    return flg, data

def clustering(data, K , attrbs):
    clust_dat = init_clustr(K,data)
    data['Clust'] = 0
    while(1):
        old_clust = data['Clust']
        flg=0
        for i in data.index:
            flg, data = clustr_assgn(data,old_clust,clust_dat,i,attrbs,flg)
        for i in clust_dat.index:
            for j in attrbs:
                clust_dat.loc[i, j] = 0
        count  = np.zeros(len(clust_dat))
        for i in data.index:
            j = data.loc[i, 'Clust']
            count[j] = count[j] + 1
            for att in attrbs:
                clust_dat.loc[j, att] += data.loc[i, att]
        for i in clust_dat.index:
            f = count[i]
            for j in attrbs:
                clust_dat.loc[i, j] /= f
        if(flg==0):
            break
    return data, clust_dat

data = pd.read_csv('Mall_Customers.csv',index_col = 'CustomerID')
labels = list(data.columns)
labels = labels[1:]
l = [[0,1],[0,2],[1,2]]
k = [3,4,5]
for i in range(0,len(l)):
    lst_attr = [labels[l[i][0]],labels[l[i][1]]]
    for j in k:
        dat_final, clust_dat = clustering(data.copy(), j, lst_attr)
        plt.scatter(dat_final['Age'], dat_final['Annual_Income'], c= dat_final['Clust'], s=40, alpha=0.5)
        plt.scatter(clust_dat.loc[:, 'Age'], clust_dat.loc[:, 'Annual_Income'], c='red', s=75)
        plt.xlabel(lst_attr[0])
        plt.ylabel(lst_attr[1])
        plt.title("Clustering for K="+ str(j)+" and parameters {"+str(lst_attr[0])+","+str(lst_attr[1])+"}")
        plt.show()