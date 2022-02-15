################# step 4 #####################
# prediction  : Pytorch Model
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
#rows is number of samples (4 sample in this case)
#columns number of features (1 feature)
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32) #domain
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32) #domain
X_test = torch.tensor([5], dtype=torch.float32)

# the shape of the tensor determines the samples and the number of features
# this case 4 samples for one feature
n_samples, n_features = X.shape #returns 4,1
print(n_samples, n_features)
input_size = n_features #1
output_size = n_features #1


#model for linear regration 
#now we dont implement W anymore
model = nn.Linear(input_size, output_size)

#loss = MSE
learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

#training
n_iter = 100

for epoch in range(n_iter):
    #prediction = forwardpass
    y_pred = model(X)

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
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}. loss= {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
