#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pickle 

# In[2]:

class Model():

    def __init__(self, x, hSize):
        import pickle
        import numpy as np
        self.w_dict = {}
        self.w_dict["a0"] = x
        self.w_dict["x"] = x
        self.hSize = hSize
        reduced_w = 0.005
        for i in range(np.shape(hSize)[1] + 1):
            
            
            
            i+= 1
            if i ==1:
                self.w_dict["w%s" %(i)] = np.random.randn(hSize[0][i-1], np.shape(x)[0]) /10
                self.w_dict["b%s" %(i)] = np.zeros((hSize[0, i -1], 1))

                self.w_dict["dw%s" %(i)] = np.zeros((hSize[0][i-1], np.shape(x)[0])) 
                self.w_dict["db%s" %(i)] = np.zeros((hSize[0, i -1], 1))
                
                self.w_dict["V" + str(i)] = np.zeros((np.shape(self.w_dict["w%s" %(i)])))
                self.w_dict["S" + str(i)] = np.zeros((np.shape(self.w_dict["w%s" %(i)])))
                self.w_dict["Vb" + str(i)] = np.zeros((np.shape(self.w_dict["b%s" %(i)])))
                self.w_dict["Sb" + str(i)] = np.zeros((np.shape(self.w_dict["b%s" %(i)])))

                
                self.w_dict["update_weights_value" + str(i)] = np.zeros((np.shape(self.w_dict["w%s" %(i)])))  
                self.w_dict["update_bias_value" + str(i)] = np.zeros((np.shape(self.w_dict["b%s" %(i)])))
                

            elif i != np.shape(hSize)[1] + 1:
                self.w_dict["w%s" %(i)] = np.random.randn(hSize[0][i-1], hSize[0][i-2]) / 10
                self.w_dict["b%s" %(i)] = np.zeros((hSize[0, i -1], 1))
                
                self.w_dict["dw%s" %(i)] = np.zeros((hSize[0][i-1], hSize[0][i-2]))
                self.w_dict["db%s" %(i)] = np.zeros((hSize[0, i -1], 1))            
                
                self.w_dict["V" + str(i)] = np.zeros((np.shape(self.w_dict["w%s" %(i)])))
                self.w_dict["S" + str(i)] = np.zeros((np.shape(self.w_dict["w%s" %(i)])))
                self.w_dict["Vb" + str(i)] = np.zeros((np.shape(self.w_dict["b%s" %(i)])))
                self.w_dict["Sb" + str(i)] = np.zeros((np.shape(self.w_dict["b%s" %(i)])))

                self.w_dict["update_weights_value" + str(i)] = np.zeros((np.shape(self.w_dict["w%s" %(i)])))  
                self.w_dict["update_bias_value" + str(i)] = np.zeros((np.shape(self.w_dict["b%s" %(i)])))
                  

            else:
                self.w_dict["w%s" %(i)] = np.random.randn(1, hSize[0][i-2])/10
                self.w_dict["b%s" %(i)] = np.zeros((1, 1))
                
                self.w_dict["dw%s" %(i)] = np.zeros((1, hSize[0][i-2])) 
                self.w_dict["db%s" %(i)] = np.zeros((1, 1))
                
                self.w_dict["V" + str(i)] = np.zeros((np.shape(self.w_dict["w%s" %(i)])))
                self.w_dict["S" + str(i)] = np.zeros((np.shape(self.w_dict["w%s" %(i)])))
                
                self.w_dict["Vb" + str(i)] = np.zeros((np.shape(self.w_dict["b%s" %(i)])))
                self.w_dict["Sb" + str(i)] = np.zeros((np.shape(self.w_dict["b%s" %(i)])))

                self.w_dict["update_weights_value" + str(i)] = np.zeros((np.shape(self.w_dict["w%s" %(i)])))  
                self.w_dict["update_bias_value" + str(i)] = np.zeros((np.shape(self.w_dict["b%s" %(i)])))
      


       # return self.w_dict


    # In[3]:

    def feedForwardGeneral(self, x, w_dict):
        for i in range(np.shape(self.hSize)[1] + 1):
            i+= 1
            if i ==1:

                self.w_dict["z%s" %(i)] = np.dot(self.w_dict["w%s" %(i)], x) + self.w_dict["b%s" %(i)]
                self.w_dict["a%s" %(i)] = np.maximum(0, self.w_dict["z%s" %(i)])
                

            elif i != np.shape(self.hSize)[1] + 1:
                self.w_dict["z%s" %(i)] = np.dot(self.w_dict["w%s" %(i)], self.w_dict["a%s" %(i-1)]) + self.w_dict["b%s" %(i)]
                self.w_dict["a%s" %(i)] = np.maximum(0, self.w_dict["z%s" %(i)])            
            else:
                self.w_dict["z%s" %(i)] = np.dot(self.w_dict["w%s" %(i)], self.w_dict["a%s" %(i-1)]) + self.w_dict["b%s" %(i)]
                self.w_dict["a%s" %(i)] =  self.Sigmoid(self.w_dict["z%s" %(i)]) 
                self.w_dict["yHat"] = self.w_dict["a%s" %(i)]
        return self.w_dict


    def feedForward(self, x,  miniBatching = False, miniBatchSize = 628):
        #print(miniBatching)
        if miniBatching == False:
            self.w_dict = self.feedForwardGeneral(x, self.w_dict)
        
        return self.w_dict


    # In[4]:


    def Cost(yHat, y):
        m = y.shape[1]
        J = -1 / m * np.sum(np.multiply(np.log(yHat), y) + np.multiply(np.log(1.001 - yHat), (1 - y))) 
        return J


    # In[5]:


    def Sigmoid(self, x):
        sig = 1/(1 + np.exp(-x))
        return sig


    # In[6]:


    def Backpropagation(self, x_train,y_train, arch, params0, numIter):
        for i in range(numIter):
            self.w_dict = feedForward(x_train, params0, arch)
            params = UpdateParameters(self.w_dict, y_train, 0.001, 2)
            cost = Cost(params["yHat"], y_train)
            #print(params["dw1"][1,1])
            print(cost)
        return params


    # In[ ]:


    def RelDer(x):
        der = np.greater(x, 0)
        return der


    def UpdateParameters(self, y, learning_rate, number_of_layers, beta1 = .9, beta2 = .99, Adam = True, Regularization = True, lmbd = 0.1, miniBatching = False, miniBatchSize = 628):
        if miniBatching ==False:
            if (Regularization != True):
                lmbd = 0
            
            update_weights_value = 0
            update_bias_value = 0
            m = self.w_dict["a0"].shape[1]
            for i in reversed(range(number_of_layers+1)):
                i+= 1
                
                
                
                if Adam:
                    self.w_dict["V" + str(i)] = beta1 * self.w_dict["V" + str(i)] + (1 - beta1) * self.w_dict["dw" + str(i)]
                    self.w_dict["S" + str(i)] = np.sqrt(beta2 * self.w_dict["S" + str(i)] + (1 - beta2) * np.square(self.w_dict["dw" + str(i)]))
                    self.w_dict["update_weights_value" + str(i)] = np.divide(self.w_dict["V" + str(i)], self.w_dict["S" + str(i)] + 0.001)

                    self.w_dict["Vb" + str(i)] = beta1 * self.w_dict["Vb" + str(i)] + (1 - beta1) * self.w_dict["db" + str(i)]
                    self.w_dict["Sb" + str(i)] = np.sqrt(beta2 * self.w_dict["Sb" + str(i)] + (1 - beta2) * np.square(self.w_dict["db" + str(i)]))
                    self.w_dict["update_bias_value" + str(i)] = np.divide(self.w_dict["Vb" + str(i)], self.w_dict["Sb" + str(i)] + 0.001)
                else:
                    self.w_dict["update_weights_value" + str(i)] = self.w_dict["dw%s" %(i)]
                    self.w_dict["update_bias_value" + str(i)] = self.w_dict["db" + str(i)]
                
                #print("V = ", self.w_dict["V" + str(i)].shape)
                #print("S = ", self.w_dict["S" + str(i)].shape)
                #print("update = ", self.w_dict["update_weights_value" + str(i)].shape)
          
                
                
                
                
                
                if(i == number_of_layers +1):
                    self.w_dict["dz%s" %(i)]  = self.w_dict["yHat"] - y
                else:
                    self.w_dict["dz%s"  %(i)] = np.multiply(np.dot(self.w_dict["w%s"%(i+1)].T, self.w_dict["dz%s"  %(i+1)]), RelDer(self.w_dict["z%s"  %(i)]))

                self.w_dict["dw%s"  %(i)] = (np.dot(self.w_dict["a%s" %(i-1)], self.w_dict["dz%s" %(i)].T)).T / m
                self.w_dict["db%s"  %(i)] = np.sum(self.w_dict["dz%s"  %(i)], axis = 1, keepdims = True) / m
                
                
                self.w_dict["w%s" %(i)] = self.w_dict["w%s" %(i)] - learning_rate * self.w_dict["update_weights_value" + str(i)] -(lmbd / m) * self.w_dict["w%s" %(i)]
                self.w_dict["b%s" %(i)] = self.w_dict["b%s" %(i)] - learning_rate * self.w_dict["update_bias_value" + str(i)]
                
               # print("reg: ", str((lmbd / m ) * np.sum(self.w_dict["w" + str(i)])))
            return self.w_dict
    def Load(self, location):
        with open(location, "br") as f:
            self.w_dict = pickle.load(f)
