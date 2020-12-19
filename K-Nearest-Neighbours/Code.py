import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

def euclidean_distance(x_test, x_train):
    distance = 0
    for i in range(len(x_test)-1):
        distance += (x_test[i]-x_train[i])**2
    return sqrt(distance)

def euclidean_norm_distance(x_test, x_train):
    distance = 0
    xt = np.linalg.norm(x_test)
    xtr = np.linalg.norm(x_train)
    for i in range(len(x_test)-1):
        distance += ((x_test[i]/xt)-(x_train[i]/xtr))**2
    return sqrt(distance)

def cosine_distance(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return (1-cos)

def get_neighbors(x_test, x_train, num_neighbors,d):
    distances = []
    data = []
    for i in x_train:
        if d=='Euclidean':
            distances.append(euclidean_distance(x_test,i))
        if d=='Normalized Euclidean':
            distances.append(euclidean_norm_distance(x_test,i))
        if d=='Cosine Similarity':
            distances.append(cosine_distance(x_test,i))
        data.append(i)
    distances = np.array(distances)
    data = np.array(data)
    sort_indexes = distances.argsort()
    data = data[sort_indexes]
    return data[:num_neighbors]

def prediction(x_test, x_train, num_neighbors, d):
    classes = []
    neighbors = get_neighbors(x_test, x_train, num_neighbors,d)
    for i in neighbors:
        classes.append(i[-1])
    predicted = max(classes, key=classes.count)              #taking the most repeated class
    return int(predicted)

def accuracy(y_true, y_pred):
    num_correct = 0
    for i in range(0,len(y_true)):
        if int(y_true[i])==int(y_pred[i]):
            num_correct+=1
    accuracy = num_correct/len(y_true)
    return accuracy

dataset = pd.read_csv("./cancer_dataset.csv")
dataset = dataset.sample(frac=1).reset_index(drop=True) # Shuffle
dataset.drop(['Index'],axis=1,inplace=True)
train_size = int(dataset.shape[0]*0.8)
train_df = dataset.iloc[:train_size,:] 
test_df = dataset.iloc[train_size:,:]
train = train_df.values
test = test_df.values
y_true = test[:,-1]
# print('Train_Shape: ',train_df.shape)
# print('Test_Shape: ',test_df.shape)
dist = ['Euclidean','Normalized Euclidean','Cosine Similarity']
p = [1,3,5,7]
for k in p:
    Acc = []
    for d in dist:
        y_pred = []
        for i in test:
            y_pred.append(prediction(i, train, k, d))
        Acc.append(accuracy(y_true, y_pred))
        print("Accuracy when K =",k," and distance metric is ",d,"=",Acc[-1])
    plt.figure()
    plt.bar(dist,Acc)
    plt.xlabel('Distance Metric Used')
    plt.ylabel('Accuracy')
    plt.title('KNN for K='+str(k))
    plt.show()