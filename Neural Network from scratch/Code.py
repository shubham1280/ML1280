import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def InitializeWeights(nodes):
    """Initialize weights with random values in [-1, 1] (including bias)"""
    layers, weights = len(nodes), []
    for i in range(1, layers):
        w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)]
              for j in range(nodes[i])]
        weights.append(np.matrix(w))
    return weights

def ForwardPropagation(x, weights, layers):
    activations, layer_input = [x], x
    for j in range(layers):
        activation = Sigmoid(np.dot(layer_input, weights[j].T))
        activations.append(activation)
        layer_input = np.append(1, activation) # Augment with bias
    return activations

def BackPropagation(y, activations, weights, layers):
    outputFinal = activations[-1]
    error = np.matrix(y - outputFinal) # Error at output
    for j in range(layers, 0, -1):
        currActivation = activations[j]
        if(j > 1):
            # Augment previous activation
            prevActivation = np.append(1, activations[j-1])
        else:
            # First hidden layer, prevActivation is input (without bias)
            prevActivation = activations[0]
        delta = np.multiply(error, SigmoidDerivative(currActivation))
        weights[j-1] += lr * np.multiply(delta.T, prevActivation)
        w = np.delete(weights[j-1], [0], axis=1) # Remove bias from weights
        error = np.dot(delta, w) # Calculate error for current layer
    
    return weights

def Train(X, Y, lr, weights):
    layers = len(weights)
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x = np.matrix(np.append(1, x)) # Augment feature vector
        activations = ForwardPropagation(x, weights, layers)
        weights = BackPropagation(y, activations, weights, layers)

    return weights

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SigmoidDerivative(x):
    return np.multiply(x, 1-x)

def Predict(item, weights):
    layers = len(weights)
    item = np.append(1, item) # Augment feature vector
    ##_Forward Propagation_##
    activations = ForwardPropagation(item, weights, layers)
    outputFinal = activations[-1].A1
    index = FindMaxActivation(outputFinal)
    # Initialize prediction vector to zeros
    y = [0 for i in range(len(outputFinal))]
    y[index] = 1  # Set guessed class to 1
    return y,outputFinal # Return prediction vector

def FindMaxActivation(output):
    """Find max activation in output"""
    m, index = output[0], 0
    for i in range(1, len(output)):
        if(output[i] > m):
            m, index = output[i], i
    return index

def Accuracy(X, Y, weights):
    """Run set through network, find overall accuracy"""
    correct = 0
    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        guess,sigmoid = Predict(x, weights)
        if(y == guess):
            # Guessed correctly
            correct += 1
    return correct*100 / len(X)

def CalculateCost(X,Y,weights):
    cost = 0
    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        guess,sigmoid = Predict(x, weights)
        if(y!=guess):
            t = np.array(y)-np.array(guess)
            cost += 0.5*(np.dot(t,t))
    return cost

def NeuralNetwork(X_train, Y_train, epochs=10, nodes=[], lr=0.15):
    hidden_layers = len(nodes) - 1
    weights = InitializeWeights(nodes)
    cost_list = list()
    epoch_list = list()
    for epoch in range(1, epochs+1):
        weights = Train(X_train, Y_train, lr, weights)

        if(epoch % 10 == 0):
            print("Epoch {}".format(epoch))
            print("Training Accuracy:{}%".format(Accuracy(X_train, Y_train, weights)))
            print("Testing Accuracy: {}%".format(Accuracy(X_test, Y_test, weights)))
            print("Cost:{}".format(CalculateCost(X_train, Y_train, weights)))
            cost_list.append(CalculateCost(X_train, Y_train, weights))
            epoch_list.append(epoch)
    plt.plot(epoch_list,cost_list)
    plt.title("Cost vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()
    return weights

def column(matrix, i):
    return [row[i] for row in matrix]

iris = pd.read_csv("./Iris.csv")
iris = iris.sample(frac=1).reset_index(drop=True) # Shuffle
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
labels = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for l in labels:
    plt.hist(iris[l])
    plt.xlabel(l)
    plt.ylabel("Count")
    plt.title("Histogram for "+l)
    plt.show()
X = np.array(X)
minmax = dataset_minmax(X)
normalize_dataset(X,minmax)
for l in range(0,len(labels)):
    plt.hist(column(X,l))
    plt.xlabel(labels[l])
    plt.ylabel("Count")
    plt.title("Histogram for "+labels[l]+" after normalization")
    plt.show()

encode_dict = {'Iris-virginica': np.array([0., 0., 1.]),'Iris-setosa': np.array([1., 0., 0.]),'Iris-versicolor': np.array([0., 1., 0.])}
Y = list()
for i in range(0,len(iris.Species)):
    Y.append(list())
for i in range(0,len(iris.Species)):
    Y[i] = encode_dict[iris.Species[i]]

### Train Test Split 80-20
train_pct_index = int(0.8 * len(X))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
Y_train, Y_test = Y[:train_pct_index], Y[train_pct_index:]

f = len(X[0]) # Number of features
o = len(Y[0]) # Number of outputs / classes

layers = [f,3,o] # Number of nodes in layers input, hidden, output
lr, epochs = 0.15, 100

weights = NeuralNetwork(X_train, Y_train, epochs=epochs, nodes=layers, lr=lr)
print("Final Testing Accuracy: {}%".format(Accuracy(X_test, Y_test, weights)))
Test_Data = [[4.6,3.5,1.8,0.2],[5.9,2.5,1.6,1.6],[5,4.2,3.7,0.3],[5.7,4,4.2,1.2]]
Normalized_test = [[4.6,3.5,1.8,0.2],[5.9,2.5,1.6,1.6],[5,4.2,3.7,0.3],[5.7,4,4.2,1.2]]
normalize_dataset(Normalized_test,minmax)
for i in range(0,len(Normalized_test)):
    j = Normalized_test[i]
    g,s = Predict(j, weights)
    for k in s:
        k = "{:.5f}".format(float(k))
    sp = ""
    for key,value in encode_dict.items():
        if list(value)==list(g):
            sp = key
    print("Predicted output value for the input",Test_Data[i],":",s,end=" ")
    print("Predicted Species:",sp)