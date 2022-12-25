#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:


def instantiate(x, hSize):
    w_dict = {}
    w_dict["a0"] = x
    w_dict["x"] = x
    reduced_w = 0.005
    for i in range(np.shape(hSize)[1] + 1):
        
        
        
        i+= 1
        if i ==1:
            w_dict["w%s" %(i)] = np.random.randn(hSize[0][i-1], np.shape(x)[0]) /100
            w_dict["b%s" %(i)] = np.zeros((hSize[0, i -1], 1))

            w_dict["dw%s" %(i)] = np.zeros((hSize[0][i-1], np.shape(x)[0])) 
            w_dict["db%s" %(i)] = np.zeros((hSize[0, i -1], 1))
            
            w_dict["V" + str(i)] = np.zeros((np.shape(w_dict["w%s" %(i)])))
            w_dict["S" + str(i)] = np.zeros((np.shape(w_dict["w%s" %(i)])))
            w_dict["Vb" + str(i)] = np.zeros((np.shape(w_dict["b%s" %(i)])))
            w_dict["Sb" + str(i)] = np.zeros((np.shape(w_dict["b%s" %(i)])))

            
            w_dict["update_weights_value" + str(i)] = np.zeros((np.shape(w_dict["w%s" %(i)])))  
            w_dict["update_bias_value" + str(i)] = np.zeros((np.shape(w_dict["b%s" %(i)])))
            

        elif i != np.shape(hSize)[1] + 1:
            w_dict["w%s" %(i)] = np.random.randn(hSize[0][i-1], hSize[0][i-2]) / 100
            w_dict["b%s" %(i)] = np.zeros((hSize[0, i -1], 1))
            
            w_dict["dw%s" %(i)] = np.zeros((hSize[0][i-1], hSize[0][i-2]))
            w_dict["db%s" %(i)] = np.zeros((hSize[0, i -1], 1))            
            
            w_dict["V" + str(i)] = np.zeros((np.shape(w_dict["w%s" %(i)])))
            w_dict["S" + str(i)] = np.zeros((np.shape(w_dict["w%s" %(i)])))
            w_dict["Vb" + str(i)] = np.zeros((np.shape(w_dict["b%s" %(i)])))
            w_dict["Sb" + str(i)] = np.zeros((np.shape(w_dict["b%s" %(i)])))

            w_dict["update_weights_value" + str(i)] = np.zeros((np.shape(w_dict["w%s" %(i)])))  
            w_dict["update_bias_value" + str(i)] = np.zeros((np.shape(w_dict["b%s" %(i)])))
              

        else:
            w_dict["w%s" %(i)] = np.random.randn(1, hSize[0][i-2])/100
            w_dict["b%s" %(i)] = np.zeros((1, 1))
            
            w_dict["dw%s" %(i)] = np.zeros((1, hSize[0][i-2])) 
            w_dict["db%s" %(i)] = np.zeros((1, 1))
            
            w_dict["V" + str(i)] = np.zeros((np.shape(w_dict["w%s" %(i)])))
            w_dict["S" + str(i)] = np.zeros((np.shape(w_dict["w%s" %(i)])))
            
            w_dict["Vb" + str(i)] = np.zeros((np.shape(w_dict["b%s" %(i)])))
            w_dict["Sb" + str(i)] = np.zeros((np.shape(w_dict["b%s" %(i)])))

            w_dict["update_weights_value" + str(i)] = np.zeros((np.shape(w_dict["w%s" %(i)])))  
            w_dict["update_bias_value" + str(i)] = np.zeros((np.shape(w_dict["b%s" %(i)])))
  


    return w_dict


# In[3]:


def feedForward(x, w_dict, hSize):
    for i in range(np.shape(hSize)[1] + 1):
        i+= 1
        if i ==1:

            w_dict["z%s" %(i)] = np.dot(w_dict["w%s" %(i)], x) + w_dict["b%s" %(i)]
            w_dict["a%s" %(i)] = np.maximum(0, w_dict["z%s" %(i)])
            

        elif i != np.shape(hSize)[1] + 1:
            w_dict["z%s" %(i)] = np.dot(w_dict["w%s" %(i)], w_dict["a%s" %(i-1)]) + w_dict["b%s" %(i)]
            w_dict["a%s" %(i)] = np.maximum(0, w_dict["z%s" %(i)])            
        else:
            w_dict["z%s" %(i)] = np.dot(w_dict["w%s" %(i)], w_dict["a%s" %(i-1)]) + w_dict["b%s" %(i)]
            w_dict["a%s" %(i)] =  Sigmoid(w_dict["z%s" %(i)])  
            w_dict["yHat"] = w_dict["a%s" %(i)]
    
    return w_dict


# In[4]:


def Cost(yHat, y):
    m = y.shape[1]
    J = -1 / m * np.sum(np.multiply(np.log(yHat), y) + np.multiply(np.log(1.001 - yHat), (1 - y))) 
    return J


# In[5]:


def Sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig


# In[6]:


def Backpropagation(x_train,y_train, arch, params0, numIter):
    for i in range(numIter):
        w_dict = feedForward(x_train, params0, arch)
        params = UpdateParameters(w_dict, y_train, 0.001, 2)
        cost = Cost(params["yHat"], y_train)
        #print(params["dw1"][1,1])
        print(cost)
    return params


# In[ ]:


def RelDer(x):
    der = np.greater(x, 0)
    return der


def UpdateParameters(w_dict, y, learning_rate, number_of_layers, beta1 = .9, beta2 = .99, Adam = True, Regularization = True, lmbd = 0.1):
    
    
    if (Regularization != True):
        lmbd = 0
    
    update_weights_value = 0
    update_bias_value = 0
    m = w_dict["a0"].shape[1]
    for i in reversed(range(number_of_layers+1)):
        i+= 1
        
        
        
        if Adam:
            w_dict["V" + str(i)] = beta1 * w_dict["V" + str(i)] + (1 - beta1) * w_dict["dw" + str(i)]
            w_dict["S" + str(i)] = np.sqrt(beta2 * w_dict["S" + str(i)] + (1 - beta2) * np.square(w_dict["dw" + str(i)]))
            w_dict["update_weights_value" + str(i)] = np.divide(w_dict["V" + str(i)], w_dict["S" + str(i)] + 0.001)

            w_dict["Vb" + str(i)] = beta1 * w_dict["Vb" + str(i)] + (1 - beta1) * w_dict["db" + str(i)]
            w_dict["Sb" + str(i)] = np.sqrt(beta2 * w_dict["Sb" + str(i)] + (1 - beta2) * np.square(w_dict["db" + str(i)]))
            w_dict["update_bias_value" + str(i)] = np.divide(w_dict["Vb" + str(i)], w_dict["Sb" + str(i)] + 0.001)
        else:
            w_dict["update_weights_value" + str(i)] = w_dict["dw%s" %(i)]
            w_dict["update_bias_value" + str(i)] = w_dict["db" + str(i)]
        
        #print("V = ", w_dict["V" + str(i)].shape)
        #print("S = ", w_dict["S" + str(i)].shape)
        #print("update = ", w_dict["update_weights_value" + str(i)].shape)
  
        
        
        
        
        
        if(i == number_of_layers +1):
            w_dict["dz%s" %(i)]  = w_dict["yHat"] - y
        else:
            w_dict["dz%s"  %(i)] = np.multiply(np.dot(w_dict["w%s"%(i+1)].T, w_dict["dz%s"  %(i+1)]), RelDer(w_dict["z%s"  %(i)]))

        w_dict["dw%s"  %(i)] = (np.dot(w_dict["a%s" %(i-1)], w_dict["dz%s" %(i)].T)).T / m
        w_dict["db%s"  %(i)] = np.sum(w_dict["dz%s"  %(i)], axis = 1, keepdims = True) / m
        
        
        w_dict["w%s" %(i)] = w_dict["w%s" %(i)] - learning_rate * w_dict["update_weights_value" + str(i)] -(lmbd / m) * w_dict["w%s" %(i)]
        w_dict["b%s" %(i)] = w_dict["b%s" %(i)] - learning_rate * w_dict["update_bias_value" + str(i)]
        
        print("reg: ", str((lmbd / m ) * np.sum(w_dict["w" + str(i)])))
    return w_dict

