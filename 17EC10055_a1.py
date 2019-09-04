#Shubham Maheshwari
#17EC10055
#Machine Learning: Programming Assignment 1: Decision Trees

import numpy as np

my_data = np.genfromtxt('data1_19.csv', delimiter=',',dtype='U')
my_data = my_data[1:,:]
fet=['pclass','age','gender']
def entropy(target_col):
    #Calculate the entropy. The only parameter of this function is the target_col parameter which specifies the target column
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data,split_attribute_name):
    """
    Calculate the information gain of a dataset. This function takes two parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    """    
    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[1:,3])
    d = data[1:,split_attribute_name]

    #Calculate the values and the corresponding counts for the split attribute 
    vals,counts= np.unique(d,return_counts=True)
    
    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data[data[:,split_attribute_name]==vals[i]][:,3]) for i in range(len(vals))])
    
    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def ID3(data,features,parent_node_class = None): 
    """
    ID3 Algorithm: This function takes three paramters:
    1. data = the data for which the ID3 algorithm should be run --> In the first run this equals the total dataset
    2. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset --> Splitting at each node
    3. parent_node_class = This is the value or class of the mode target feature value of the parent node for a specific node. This is 
    also needed for the recursive call since if the splitting leads to a situation that there are no more features left in the feature
    space, we want to return the mode target feature value of the direct parent node.
    """   
    #Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#
    #If all target_values have the same value, return this value
    if len(np.unique(data[:,3])) <= 1:
        return np.unique(data[:,3])[0]
    
    #If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data)==0:
        return np.unique(my_data[:,3])[np.argmax(np.unique(my_data[:,3],return_counts=True)[1])]
    elif len(features) ==0:
        return parent_node_class
    
    #If none of the above holds true, grow the tree!
    else:
        #Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[:,3])[np.argmax(np.unique(data[:,3],return_counts=True)[1])]
        
        #Select the feature which best splits the dataset
        item_values = [InfoGain(data,feature) for feature in features] #Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        #gain in the first run
        tree = {fet[best_feature]:{}}
        
        #Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]
        
        #Grow a branch under the root node for each possible value of the root node feature
        for value in np.unique(data[:, best_feature]):
            value = value
            #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data[data[:, best_feature] == value]
            
            #Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = ID3(sub_data,features,parent_node_class)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[fet[best_feature]][value] = subtree

        return(tree)

tree = ID3(my_data,[0,1,2])
#format the tree for printing
def formatData(t,s):
    if not isinstance(t,dict) and not isinstance(t,list):
        print ("\t"*s+str(t))
    else:
        for key in t:
            print ("\t"*s+str(key))
            if not isinstance(t,list):
                formatData(t[key],s+1)

formatData(tree,0)