#Shubham Maheshwari
#17EC10055
#Machine Learning: Programming Assignment 2: Naive Bayes Classifier

import numpy as np
import pandas as pd

df = pd.read_csv('data2_19.csv')
string = df.columns[0]
string = string.split(',')
dat = []
for i in range(len(df)):
    dat.append(df['D,X1,X2,X3,X4,X5,X6'][i].split(','))

df = pd.DataFrame(dat,columns=string)

#probabilities of target variable's classes in training set
t = dict(df['D'].value_counts())
t3 = list(t.values())
sm = np.sum(t3)
probD = []
for i in sorted(t.keys()):
    probD.append(t.get(i)/sm)

dictionary = dict.fromkeys(df.columns[1:])    

#Training(storing probability values using training set given)
for o in df.columns[1:]:
    r = dict(df[o].value_counts())
    prob = {'1':[],'2':[],'3':[],'4':[],'5':[]}
    for j in sorted(prob.keys()):
        if j not in r.keys():
            prob.get(j).append(1/(t['0']+2))
            prob.get(j).append(1/(t['1']+2))
        else:
            count=0
            for i in range(len(df)):
                if(df[o][i]==j and df['D'][i]=='0'):
                    count=count+1
            if(count!=0):
                prob.get(j).append((count+1)/(t['0']+5))
            else:
                prob.get(j).append(1/(t['0']+5))
            if((r.get(j)-count)!=0):
                prob.get(j).append((r.get(j)-count+1)/(t['1']+5))
            else:
                prob.get(j).append(1/(t['1']+5))
    dictionary[o] = prob
#print(dictionary)
#Load test set
dftest = pd.read_csv('test2_19.csv')
string = dftest.columns[0]
string = string.split(',')
datest = []
for i in range(len(dftest)):
    datest.append(dftest['D,X1,X2,X3,X4,X5,X6'][i].split(','))
    

dftest = pd.DataFrame(datest,columns=string)
actual = list(dftest['D'].astype(int))
dftest.drop(['D'],inplace=True,axis=1)
#make predictions
y = []
for i in dftest.index:
    t0 = probD[0]
    t1 = probD[1]
    for j in dftest.columns:
            er = dictionary.get(j).get(dftest[j][i])
            t0 = t0*er[0]
            t1 = t1*er[1]
    if(t0>t1):
        y.append(0)
    else:
        y.append(1)
# print(y)
# print(actual)
#accuracy printing
count1 = 0
for i in range(len(y)):
    if(y[i]==actual[i]):
        count1=count1+1
#print("You got",count1,"out of 14 correct")
print("Test Accuracy:",count1*100/len(y),"%")