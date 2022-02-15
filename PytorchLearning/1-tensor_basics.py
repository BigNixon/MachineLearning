import torch
import numpy as np

# ==============Tensor basics==============
x = torch.rand(2,2) #random elements
print(x)

x = torch.empty(3) #empty tensor of 3 elements
print(x)

x = torch.zeros(2,2) # 2x2 tensor with zeros
print(x)

y = torch.rand(2,2)
print(y)

##============== operation of tensors==============
x.add_(y)  #saves in the same variable
print(x)

z = torch.mul(x,y)
print(z)

##slicing of tensors
print(z[:,0]) 
print(z[1,1]) #tensor 1x1
print(z[1,1].item()) #shows the number inside of the 1x1 tensor

#==============reshaping a tensor==============
x = torch.rand(4,4)
print(x)

y = x.view(16) #reshapes x in 1x16
print(y)
y = x.view(-1,8) # determines automatically the first dimension
print(y)



## ==============woriking with numpy==========
print("## woriking with numpy")
a = torch.ones(5)
print(a)
b = a.numpy() #but aims to the same memory location
print(b)


a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b) #numpy only suports vectors in cpu

##============== creating tensors in the GPU==============
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device) #creates on gpu
    y = torch.ones(5)   #creates on cpu
    y = y.to(device)    # moves to GPU
    z = x+y             #operates in gpu
    z = z.to("cpu")     #moves to cpu so we can work with numpy

## to work with the gradients 
x = torch.ones(5, requires_grad = True)
print(x)

