import time
import math
import random
#Creating Matrix#
def createMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m  

#Predict Class
class Predict:
    def __init__(self, ni, nh, no,lr):
        # number of input, hidden, and output nodes
        self.ninput= int(ni) 
        self.nhidden= int(nh)
        self.noutput = int(no)
        self.lrate=float(lr)
        # activations for input, hidden and output nodes
        self.iactive = [1.0]*self.ninput
        self.hactive = [1.0]*self.nhidden
        self.oactive = [1.0]*self.noutput
        # create weights from input to hidden and weights from hidden to output nodes
        self.iweight = createMatrix(self.ninput, self.nhidden)
        self.oweight = createMatrix(self.nhidden, self.noutput)
        # Assignning Bias Value
        self.bias=1.0  
        # Assigning random values for  weights    
        for i in range(self.ninput):
            for j in range(self.nhidden):
                self.iweight[i][j] = random.uniform(0, 1)
        for j in range(self.nhidden):
            for k in range(self.noutput):
                self.oweight[j][k] = random.uniform(0, 1)
## Implementing Feedforward network
    def update(self, inputs):  
        #Normalizing input values
        for i in range(0,self.ninput):
            self.iactive[i] = float(inputs[i])/float(255)
        #Finding activations for hidden layer 
        for j in range(0,self.nhidden):
            sum = 0.0
            for i in range(0,self.ninput):
                sum +=( self.iactive[i] * self.iweight[i][j] )
            sum+=self.bias    
            self.hactive[j] =sum
          
        t=max(self.hactive,key=float)  
        r=min(self.hactive,key=float)
        #Normalizing activations for hidden layer
        for i in range(0,self.nhidden):
            self.hactive[i]=(self.hactive[i]-r)/(t-r)
            self.hactive[i]=1/(1+math.exp(-self.hactive[i]))
        #Finding activations for output nodes
        for k in range(0,self.noutput):
            sum = 0.0
            for j in range(0,self.nhidden):        
                sum +=( self.hactive[j] * self.oweight[j][k] )   
            self.oactive[k] = sum +self.bias  
        t=max(self.hactive,key=float)  
        r=min(self.hactive,key=float)
        #Normalizing activations for output layer
        for i in range(0,self.noutput):
            self.oactive[i]=(self.oactive[i]-r)/(t-r)
            self.oactive[i]=1/(1+math.exp(-self.oactive[i]))
        #Returns the computed orientation    
        t=max(self.oactive,key=float) 
        return self.oactive.index(t)*90
    def backpropagate(self, targets,lrate):
        #Compute output deltas 
        odeltas = [0.0] * self.noutput
        for k in range(self.noutput):
            error = targets[k] - self.oactive[k]
            odeltas[k] =  error * self.oactive[k]*(1-self.oactive[k])
        #Change output weights    
        for j in range(self.nhidden):
            for k in range(self.noutput):
                change = odeltas[k] * self.hactive[j]
                self.oweight[j][k] += lrate*change
        hdeltas = [0.0] * self.nhidden
        #Compute hidden deltas
        for j in range(self.nhidden):
            error = 0.0
            for k in range(self.noutput):
                error += odeltas[k] * self.oweight[j][k]
            hdeltas[j] = error * self.hactive[j]*(1-self.hactive[j])
        #Change input weights
        for i in range (self.ninput):
            for j in range (self.nhidden):
                change = hdeltas[j] * self.iactive[i]
                self.iweight[i][j] += lrate*change 

    def test(self, patterns,s):
        t=self.update(patterns) 
        if t==s:
            return s
        else:
            return -1      
     #Training the neural net       
    def train(self,train_data):
        #lrate=0.7#learning rate
        for i in range(0,len(train_data)/2):
            l=map(int,train_data[i][2:])
            s=int(train_data[i][1])
            p=[l,[s]]
            targets=[0]*4
            inputs = p[0]
            targets[s/90] = 1
            self.update(inputs)
            self.backpropagate(targets, self.lrate)
    #Testing the Samples        
    def tests(self,test_data):    
        count=0 
        Matrix = [[0 for x in range(4)] for x in range(4)] 
        fl = open("nnet_output.txt", 'w');
        for i in range(0,len(test_data)):
            l=map(int,test_data[i][2:])
            s=int(test_data[i][1])
            p=[l,[s]] 
            inputs = p[0]  
            t=self.test(inputs,s) 
            Matrix[s/90][t/90]+=1
            orientation=test_data[i][0]+" "+test_data[i][1]+"\n"
            fl.write(orientation)
            if t==s:
                count+=1
               

        print "Percentage of Efficiency",float(count)/float(len(test_data))*100  
        print "Confusion Matrix"
        for i in range(0,4):
            print Matrix[i]   
        
            
   # Finding the nearest neighbour of the data point and return the orientation  
    def knnd(self,train_data,test_data,k):
            klist= [float(100000000000000000) for x in range(0,int(k))]
            kind=[None]*int(k)
            idata=list(test_data[2:])
            for j in range(0,len(train_data)/20):
                distance=0
                jdata=list(train_data[j][2:])
                #Using Euclidean distance function
                #for k in range(0,192):
                 #   t=float(idata[k])-float(jdata[k])
                  #  distance+=float((t**2) )  
                #distance=math.sqrt(distance)
                #Using Mahattan Distance
                
                for k in range(0,192):
                    distance+=abs(float(idata[k])-float(jdata[k]))
                   
                t=max(klist,key=float)
                if distance < t :
                    ind=klist.index(t)
                    klist[ind]=float(distance)
                    kind[ind]=j 
 
            ori=list()        
            for kindl in range(0,len(kind)):
                ori.append(train_data[kind[kindl]][1])
            olist=['0','90','180','270']
            word_counter={}
            for word in ori:
                if word in word_counter:
                    word_counter[word] += 1
                else:
                    word_counter[word] = 1       
            return max(word_counter, key = word_counter.get ) 
    #KNN implemantion               
    def knn(self,train_data,test_data,k):
        count=0
        Matrix = [[0 for x in range(4)] for x in range(4)] 
        fl = open("knn_output.txt", 'w');
        #Reading each test sample to find the orientation
        for i in range(0,len(test_data)):
            t=list(test_data[i])
            orientation=self.knnd(train_data,t,k)
            if orientation == t[1]:
                count+=1  
            Matrix[int(t[1])/90][int(orientation)/90]+=1       
            orientation=t[0]+" "+orientation+"\n"
            fl.write(orientation)
        o=float(count)/float(len(test_data)) 
        print "Percentage of Efficiency",o*float(100) 
        print "Confusion Matrix"
        for i in range(0,4):
            print Matrix[i]



                        