############# step 1##################3
# all manually
# prediction
# gradient computation
# loss computation
# parameter updates
from typing import ForwardRef
import numpy as np

# f = w*x
X = np.array([1,2,3,4], dtype=np.float32) #domain
Y = np.array([2,4,6,8], dtype=np.float32) #real values

W = 0.0

#model prediction
def fordward(x):
    return W*x

#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

#gradient
#MSE = 1/N * (W*X - Y)**2
# dJ/dw = 1/N * 2X * (w*x - y)
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted - y).mean()

print(f'Prediction before training: f(5) = {fordward(5):.3f}')

#training
learning_rate = 0.01
n_iter = 100

for epoch in range(n_iter):
    #prediction = forwardpass
    y_pred = fordward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients
    dw = gradient(X,Y,y_pred)

    #update weights
    W -= learning_rate * dw

    if epoch%1 ==0:
        print(f'epoch {epoch + 1}: w = {W:.3f}. loss= {l:.8f}')

print(f'Prediction after training: f(5) = {fordward(5):.3f}')

