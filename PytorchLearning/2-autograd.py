import torch
 #calculate gradients automatically witch autograd package
x = torch.rand(3, requires_grad=True)
print(x)

y = x+2 # creates a computationla graph (each operation like nodes)
#       X ---
#            \-- (+)--- Y
#       2 ---
print(y) #grad_fn=<AddBackward0>

z = y*y*2
z = z.mean() #grad_fn=<MeanBackward0>
print(z)

z.backward() #dz/dx
print(x.grad) #prints the gradients in this tensor


z = y*y*2
print(z) #now the z is not scalar so we need to declare vector v
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v) #dz/dx
print(x.grad) #prints the gradients in this tensor


## ========= prevent of tracking the gradients====================

# x.requires_grad_(False)
# x.detach()
# with torch.nograd():

with torch.no_grad():
    y = x+2
    print(y) 

## ========== avoid to gradients to be summed ========
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() #without this, the gradient accumulates and this is incorrect

##=============using stochastic gradient descent==========
weights = torch.ones(4, requires_grad=True)
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()



