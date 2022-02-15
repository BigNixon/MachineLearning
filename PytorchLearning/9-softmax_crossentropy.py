'''
understanding axis
axis=i is the i-th index of shape


lets have the example:
        b = np.array([[1,2,3],
                    [4,5,6]])
        print(b.shape)
        #prints (   2       ,       3)
                axis = 0       axis = 1 

CASE A)
    sum = np.sum(b, axis=0) #sum = [5, 7, 9] of shape(3,)
we expect it to find two numbers along this axis (by looking at the shape). So [1+4, 2+5, 3+6]

CASE B)
    sum = np.sum(b, axis=1) #sum = [6, 15] of shape(2,)
    from the shape we can see this it is an axis along which there are 3 numbers to be summed. So, [1+2+3,4+5+6]

'''

import torch
import torch.nn as nn
import numpy as np

##################### SOFTMAX ###################
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(f'probability numpy: {outputs}')



x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0) #across the first axis
print(f'probability torch: {outputs}')  


################## CROSS ENTROPY #####################
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss #/ float(predicted[0])

#y must be one hot encoded
# class 0: [1 0 0]
# class 1: [0 1 0]
# class 2: [0 0 1]
Y = np.array([1, 0, 0])

#y_pred has probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'loss1 numpy: {l1:.4f}')
print(f'loss2 numpy: {l2:.4f}')

#nn.CrossentropyLoss already aplies 
#nn.LogSoftmax + nn.NLLLoss (negative log likehood loss)
#we dont have to put a softmax in last layer
#already has class labels => no one-hot
#=>no softmax

loss = nn.CrossEntropyLoss()

#Y = torch.tensor([0])
#n_samples x n_class = 1x3

#values before softmax
#Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
#Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

Y = torch.tensor([2, 0, 1])
#n_samples x n_class = 3x3

#values before softmax
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])


l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'loss1 torch: {l1.item():.4f}')
print(f'loss2 torch: {l2.item():.4f}')

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)