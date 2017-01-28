#####Implementation of KNN #####
#We get the train and test dataset and value of K
#We parse through each of the test sample and  find its k nearest neifghbour in training set 
#and assign orientation which has maximum count on k training samples
#We use euclidean distance to compute the distance of test sample from each of the training data


#####Implementation of Neural net #####
#We take in number of nodes in hiddenlayer as input
#We assign random weights to inputs and hidden nodes
#We assign the pixel value as input activation and compute hidden activation from input activation and input weights
#We calculate output activation from hidden weights and hidden activations
#On finding tha output weight we compute the error and do back propogation
#we calculate the output delta and update the output weights
#calculate the hidden delta and update the input weights
#On calcutaing the weights for three layers by using the train data
#we compute the orientation of test data based on the weights calculated from train data
#We normalize the values in each layer before using it in the sigmoid functions
#We use the sigmoid function y= (1/(1+e-x))
#Gradient descent y*(1-y)
#Referred the link for neural network for xor implementation http://code.activestate.com/
#Please use bestmodel.txt for running the best models##

import sys
from predict import Predict
import time
# Read in training or test data file to the variable test_data or train_data
#
def read_data(fname):
    ideal = []
    file = open(fname, 'r');
    li = [i.strip().split() for i in file.readlines()]
    ideal +=li
    return ideal
# Read in the file #
if len(sys.argv) != 5:
    print "Usage: python  orientation.py train-data.txt test-data.txt knn k"
    sys.exit()
(train_file, test_file) = sys.argv[1:3]

print "Learning images..."

predict= Predict(0,0,0,0)
train_data = read_data(train_file)
test_data = read_data(test_file)

print "Testing classifiers..."
# Calling KNN or NNET for image Classification#
if(sys.argv[3]=="knn"):
    start_time=time.time()
    predict.knn(train_data,test_data,sys.argv[4])
    print ("----Seconds----",time.time()-start_time) 
if (sys.argv[3]=="nnet"):
    start_time=time.time()
    predict= Predict(192,sys.argv[4],4,0.7)
    predict.train(train_data)
    predict.tests(test_data)
    print ("----Seconds----",time.time()-start_time)  
if (sys.argv[3]=="best"):
    with open("bestmodel.txt", "r") as ins:
        for line in ins:
            a=line.split()
            if a[0]=="nnet":
                start_time=time.time()
                predict= Predict(192,a[2],4,a[1])
                predict.train(train_data)
                predict.tests(test_data)
                print ("----Seconds----",time.time()-start_time) 
            if a[0]=="knn":
                start_time=time.time()
                predict= Predict(0,0,0,0)
                start_time=time.time()
                predict.knn(train_data,test_data,a[1])
                print ("----Seconds----",time.time()-start_time) 
            




