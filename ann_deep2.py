import numpy as np
from math import e
from random import random
from copy import deepcopy 

def sigmoid(number):
    #print(number)
    return 1.0 / (1.0 + e ** float(-number) )

def df_sigmoid(number):
    return (number * (1 - number))

class ANN:
    def __init__(self , size_input_layer , size_hiddens_layers , size_output_layer , num_hiddens_layers , activation_function=sigmoid,df_activation_function=df_sigmoid,learning_rate=0.1):
      
        self.num_hiddens_layers = num_hiddens_layers
        self.hiddens_layers = []
        self.learning_rate = learning_rate
        self.size_input_layer = size_input_layer
        self.size_hiddens_layers = size_hiddens_layers
        self.size_output_layer = size_output_layer
        self.activation_function = activation_function
        self.df_activation_function = df_activation_function

        size_previous_layer = size_input_layer
        
        for n in range(self.num_hiddens_layers):
            hl_weights = []
            for i in range(size_previous_layer):
                l = []
                for j in range(self.size_hiddens_layers):
                    c = random() * 2 - 1
                    l.append(c)
                    
                hl_weights.append(l)
                
            self.hiddens_layers.append( {'ws':np.matrix(hl_weights) , 'bs':np.matrix( [ random() * 2 - 1 for i in range(self.size_hiddens_layers) ] ) })
            size_previous_layer = self.size_hiddens_layers

            
        
        ho_weights = []
        
        for i in range(self.size_hiddens_layers):
            l = []
            for j in range(self.size_output_layer):
                c = random() * 2 - 1
                l.append(c)
                
            ho_weights.append(l)
            

        self.ho_weights = np.matrix(ho_weights)

        self.ho_bias = np.matrix( [ random() * 2 - 1 for i in range(self.size_output_layer) ] )
        
        print("number hiddens layers:",len(self.hiddens_layers))
        input("[ENTER]")

    def __feed_forward(self,inputs):

        if inputs.shape[1] == self.size_input_layer:
            outputs_hiddens_layers = []
            input_hl = inputs
            
            for n in range(self.num_hiddens_layers):
                output_hl = input_hl * self.hiddens_layers[n]['ws'] + self.hiddens_layers[n]['bs']

                
                for i in range(output_hl.shape[0]):
                    for j in range(output_hl.shape[1]):
                        output_hl[i,j] = self.activation_function(output_hl[i,j])
                
                
                #print("outputs hl%d:"%n,output_hl)
                outputs_hiddens_layers.append(deepcopy(output_hl))
                
                input_hl = deepcopy(output_hl)
                
            input_ol = deepcopy(output_hl)

            output_ol = input_ol * self.ho_weights + self.ho_bias

            for i in range(output_ol.shape[0]):
                for j in range(output_ol.shape[1]):
                    output_ol[i,j] = self.activation_function(output_ol[i,j])
                    
            #print("outputs ol:",output_ol) 
            
            return output_ol , outputs_hiddens_layers
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
        outputs_ol , outputs_hiddens = self.__feed_forward(inputs)

        #calculating errors of output layer...
        output_errors = targets - outputs_ol
        
        #calculating squared errors for return to function train...
        squared_errors = sum( [ oe ** 2 for oe in output_errors.tolist()[0] ] )
        
        #H->0 AFTER H->H OR ALWAYS H->O?
        hiddens_errors = [0 for i in range( self.num_hiddens_layers )]
        wt = self.ho_weights.getT()
        next_layer_errors = deepcopy(output_errors)
        for i in range( self.num_hiddens_layers - 1 , -1 , -1 ):
            #calculating errors of each hidden layer...
            hiddens_errors[i] =  next_layer_errors * wt
            wt = self.hiddens_layers[i]['ws'].getT()
            next_layer_errors = deepcopy(hiddens_errors[i])


        #calculating gradient for output layer...
        gradient_output = map(self.df_activation_function , np.array(outputs_ol))
        gradient_output = np.matrix(list(gradient_output))
 
        
        #hadmard products!
        gradient_output = self.__hadamard_product(gradient_output , output_errors)
        
        #scalar product (np supports it)!
        gradient_output = gradient_output * self.learning_rate
        
        #calculating gradients for each hidden layer...
        hiddens_gradients = []
        for i in range(self.num_hiddens_layers):
            gradient_hidden = map(self.df_activation_function , np.array(outputs_hiddens[i]))
            gradient_hidden = np.matrix(list(gradient_hidden))

            #hadmard products!
            gradient_hidden = self.__hadamard_product(gradient_hidden , hiddens_errors[i])

            #scalar product (np supports it)!
            gradient_hidden = gradient_hidden * self.learning_rate
            
            hiddens_gradients.append(deepcopy(gradient_hidden))

            
        #change weights from HIDDEN -> OUTPUT...
        outputs_hl_t = outputs_hiddens[-1].getT()

        delta_w_output = outputs_hl_t * gradient_output

        
        #finally change weights of output layer...
        self.ho_weights = self.ho_weights + delta_w_output
        self.ho_bias = self.ho_bias + gradient_output

        
        #change weights from HIDDEN_n -> HIDDEN_n+1 where always n > 0...
        for i in range( self.num_hiddens_layers - 1 , 0 , -1 ):
            inputs_layer_t = outputs_hiddens[i-1].getT()

            delta_w_hidden = inputs_layer_t * hiddens_gradients[i]
            
            #finally change weights of hidden layer...
            self.hiddens_layers[i]['ws'] +=  delta_w_hidden
            self.hiddens_layers[i]['bs'] += hiddens_gradients[i]
            

        #change weights from INPUT -> HIDDEN_0...
        inputs_layer_t = inputs.getT()
        delta_w_hidden = inputs_layer_t * hiddens_gradients[0]
        #finally change weights of hidden layer...
        self.hiddens_layers[0]['ws'] +=  delta_w_hidden
        self.hiddens_layers[0]['bs'] += hiddens_gradients[0]
        
        return squared_errors
    
    def resolve(self,inputs_list):
        inputs = np.matrix(inputs_list)
        outputs , _ = self.__feed_forward(inputs)
        return outputs.tolist()





if __name__ == "__main__":
    
    ann = ANN(size_input_layer=2 , size_hiddens_layers=5 , size_output_layer=2,num_hiddens_layers=1,learning_rate=0.1)
    dataset = np.matrix([[1,1,1,0],[1,0,0,1],[0,1,0,1],[0,0,1,0]])
    ann.train(dataset)
    
    while True:
        #cast to float because int generates problems with np.matrix...
        a = float(input("A: "))
        b = float(input("B: "))
    
        r = ann.resolve([a,b])
        print("response:",r)
    
        input("[ENTER]")
