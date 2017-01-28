#Predicting Orientation of pictures

Use Feed-forward Neural Network and K-nearest neighbor algorithm to predict the correct orientation of pictures.

#Dataset

Pictures are taken from flickr.com, resized to 8x8. 

The train and test pictures are written into .txt file as 192 feature vectors. For each picture the train and test file has the values of the form  

**image_id correct_orientation r11 g11 b11 r12 g12 b12 ...**  

Where   

photo id is a photo ID for the image.  

correct_orientation is 0,90,180,270  

r11 refers to the red pixel value at row 1 column 1, r12 refers to red pixel at row 1 column 2, etc  


#K- Nearest Neighbor  

1. We get the train and test dataset and value of K  
2. We parse through each of the test sample and  find its k nearest neighbour in training set  
3. and assign orientation which has maximum count on k training samples  
4. We use euclidean distance to compute the distance of test sample from each of the training data  


#Neural Network

1.  We take in number of nodes in hiddenlayer as input  
2.  We assign random weights to inputs and hidden nodes  
3.  We assign the pixel value as input activation and compute hidden activation from input activation and input weights  
4.  We calculate output activation from hidden weights and hidden activations  
5.  On finding tha output weight we compute the error and do back propogation  
6.  We calculate the output delta and update the output weights  
7.  Calculate the hidden delta and update the input weights  
8.  On calcutaing the weights for three layers by using the train data  
9.  We compute the orientation of test data based on the weights calculated from train data    
10. We normalize the values in each layer before using it in the sigmoid functions    
11. We use the sigmoid function y= (1/(1+e-x))  
12. Gradient descent y*(1-y)  

#Steps to run the code

#KNN

**python orientation.py train_file.txt test_file.txt knn k**

where  

k is an integer for number of neighbors to be considered

#Neural Network

**python orientation.py train_file.txt test_file.txt nnet hidden_count**

where  

hidden_count is number of nodes in the hidden layer



