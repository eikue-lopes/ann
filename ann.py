import numpy as np
from math import e
from random import random
from copy import deepcopy 

def sigmoid(number):
    return 1 / (1 + e ** (-number) )

def df_sigmoid(number):
    return (number * (1 - number))

class ANN:
    def __init__(self , size_input_layer , size_hidden_layer , size_output_layer , activation_function=sigmoid,df_activation_function=df_sigmoid,learning_rate=0.1):

        hl_weights = []
        for i in range(size_input_layer):
            l = []
            for j in range(size_hidden_layer):
                c = random() * 2 - 1
                l.append(c)
                
            hl_weights.append(l)
        

        ho_weights = []
        for i in range(size_hidden_layer):
            l = []
            for j in range(size_output_layer):
                c = random() * 2 - 1
                l.append(c)
                
            ho_weights.append(l)
            
        self.learning_rate = learning_rate
        self.size_input_layer = size_input_layer
        self.size_hidden_layer = size_hidden_layer
        self.size_output_layer = size_output_layer
        self.activation_function = activation_function
        self.df_activation_function = df_activation_function
        self.hl_weights = np.matrix(hl_weights)
        self.ho_weights = np.matrix(ho_weights)
        self.hl_bias = np.matrix( [ random() * 2 - 1 for i in range(self.size_hidden_layer) ] )
        self.ho_bias = np.matrix( [ random() * 2 - 1 for i in range(self.size_output_layer) ] )
        
        print("weights hidden layer:\n",self.hl_weights)
        print()
        print("weights output layer:\n",self.ho_weights)
        print()
        print("bias hidden layer:\n",self.hl_bias)
        print()
        print("bias output layer:\n",self.ho_bias)
        print()
        input()
 
    def __feed_forward(self,inputs):

        if inputs.shape[1] == self.size_input_layer:

            input_hl = inputs

            output_hl = input_hl * self.hl_weights + self.hl_bias
            
            for i in range(output_hl.shape[0]):
                for j in range(output_hl.shape[1]):
                    output_hl[i,j] = self.activation_function(output_hl[i,j])
            
            input_ol = deepcopy(output_hl)

            output_ol = input_ol * self.ho_weights + self.ho_bias

            for i in range(output_ol.shape[0]):
                for j in range(output_ol.shape[1]):
                    output_ol[i,j] = self.activation_function(output_ol[i,j])
                    
            return output_ol , output_hl
        else:
            raise Exception("Number of inputs must be equal the number of inputs of input layer!")
            exit(1)
        
    def __hadamard_product(self,m1,m2):
        if m1.shape == m2.shape:
            mr = deepcopy(m1)
            for i in range(m1.shape[0]):
                for j in range(m1.shape[1]):
                    mr[i,j] = m1[i,j] * m2[i,j]
                    
            return mr
        else:
            raise Exception("Dimensions of m1",m1.shape,"and m2",m2.shape,"must be equals in haddamard product!")
            exit(1)
    def train(self,dataset,threshold=0.001):
        squared_errors = threshold * 2
        while squared_errors > threshold:
            print("Squared Errors:",squared_errors)
            
            for i in range(dataset.shape[0]):
                inputs = dataset[i,0:self.size_input_layer]
                targets = dataset[i,self.size_input_layer:]
                #print(inputs,targets)

                squared_errors = self.__backpropagation(inputs,targets)
        
    def __backpropagation(self,inputs,targets):
        outputs_ol , outputs_hl = self.__feed_forward(inputs)

        #calculating errors of output layer...
        output_errors = targets - outputs_ol
        
        #calculating squared errors for return to function train...
        squared_errors = sum( [ oe ** 2 for oe in output_errors.tolist()[0] ] )
        
        
        whot = self.ho_weights.getT()
        
        #calculating errors of hidden layer...
        hidden_errors =  output_errors * whot

        
        #calculating gradient output...
        gradient_output = map(self.df_activation_function , np.array(outputs_ol))
        gradient_output = np.matrix(list(gradient_output))
 
        
        #hadmard products!
        gradient_output = self.__hadamard_product(gradient_output , output_errors)
        
        #scalar product (np supports it)!
        gradient_output = gradient_output * self.learning_rate

        #gradients for next layer...
        gradient_hidden = map(self.df_activation_function , np.array(outputs_hl))
        gradient_hidden = np.matrix(list(gradient_hidden))

        
        #hadmard products!
        gradient_hidden = self.__hadamard_product(gradient_hidden , hidden_errors)

        #scalar product (np supports it)!
        gradient_hidden = gradient_hidden * self.learning_rate

        
        #change weights from HIDDEN -> OUTPUT...
        outputs_hl_t = outputs_hl.getT()

        delta_w_output = outputs_hl_t * gradient_output

        
        #finally change weights of output layer...
        self.ho_weights = self.ho_weights + delta_w_output
        self.ho_bias = self.ho_bias + gradient_output

        
        #change in weights from INPUT -> HIDDEN...
        inputs_t = inputs.getT()

        delta_w_hidden = inputs_t * gradient_hidden

        
        #finally change weights of hidden layer...
        self.hl_weights = self.hl_weights + delta_w_hidden
        self.hl_bias = self.hl_bias + gradient_hidden
        
        return squared_errors
    
    def resolve(self,inputs_list):
        inputs = np.matrix(inputs_list)
        outputs , _ = self.__feed_forward(inputs)
        return outputs.tolist()





ann = ANN(size_input_layer=2 , size_hidden_layer=5 , size_output_layer=2,learning_rate=0.01)
dataset = np.matrix([[1,1,1,0],[1,0,0,1],[0,1,0,1],[0,0,1,0]])
ann.train(dataset)

while True:
    #cast to float because int generates problems with np.matrix...
    a = float(input("A: "))
    b = float(input("B: "))
    
    r = ann.resolve([a,b])
    print("response:",r)
    
    input("[ENTER]")