############### step 2 ##########################3
# prediction  : manually
# gradient computation : Autograd
# loss computation  : manually
# parameter updates : manually

import torch

# f = w*x
X = torch.tensor([1,2,3,4], dtype=torch.float32) #domain
Y = torch.tensor([2,4,6,8], dtype=torch.float32) #domain

W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model prediction
def fordward(x):
    return W*x

#loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


print(f'Prediction before training: f(5) = {fordward(5):.3f}')

#training
learning_rate = 0.01
n_iter = 100

for epoch in range(n_iter):
    #prediction = forwardpass
    y_pred = fordward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients = backwardpas
    l.backward() #dl/dw

    #update weights
    with torch.no_grad(): #this operation should not be part of our graph
        W -= learning_rate * W.grad

    #zero gradients (avoid acummulation so restar the gradient)
    W.grad.zero_()

    if epoch%1 ==0:
        print(f'epoch {epoch + 1}: w = {W:.3f}. loss= {l:.8f}')

print(f'Prediction after training: f(5) = {fordward(5):.3f}')
