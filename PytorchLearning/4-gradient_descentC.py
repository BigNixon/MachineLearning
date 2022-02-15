############## step 3 #########################
# prediction  : manually
# gradient computation : Autograd
# loss computation  : PytorchLoss
# parameter updates : pytorch optimizer

# 1) DESIGN MODEL (input, output size, fordward pass)
# 2) construct loss and optimizer
# 3) training loop
#   - fordward pass  : gradients
#   - backward pass  : gradients
#   - update weights  
import torch
import torch.nn as nn

# f = w*x
X = torch.tensor([1,2,3,4], dtype=torch.float32) #domain
Y = torch.tensor([2,4,6,8], dtype=torch.float32) #domain

W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model prediction
def fordward(x):
    return W*x

#loss = MSE
learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD([W], lr = learning_rate)


print(f'Prediction before training: f(5) = {fordward(5):.3f}')

#training
n_iter = 100

for epoch in range(n_iter):
    #prediction = forwardpass
    y_pred = fordward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients = backwardpas
    l.backward() #dl/dw

    #update weights
    optimizer.step() #updates the wights 

    #zero gradients (avoid acummulation so restar the gradient)
    #W.grad.zero_()
    optimizer.zero_grad()

    if epoch%1 ==0:
        print(f'epoch {epoch + 1}: w = {W:.3f}. loss= {l:.8f}')

print(f'Prediction after training: f(5) = {fordward(5):.3f}')
