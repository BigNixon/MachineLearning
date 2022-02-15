'''
epochs = all samples
batch_size = a piece of samples in one forward and backward pass
e.g. 100 samples , batch_size = 20 -> 5 iterations for each epoch

============ built in atributes ==========
class Building(object):
     def __init__(self, floors):
         self._floors = [None]*floors
     def __setitem__(self, floor_number, data):
          self._floors[floor_number] = data
     def __getitem__(self, floor_number):
          return self._floors[floor_number]

building1 = Building(4) # Construct a building with 4 floors
building1[0] = 'Reception'
building1[1] = 'ABC Corp'
building1[2] = 'DEF Inc'
print( building1[2] )




'''
import torch
from torch.utils import data
import torchvision
from torch.utils.data import Dataset, DataLoader, dataset
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:]) #all rows from column 1-end
        self.y = torch.from_numpy(xy[:, [0]])#classes are in the first column
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
#first_data = dataset[0]
#features, labels = first_data
#print(features,labels)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
#dataiter = iter(dataloader)
#data = dataiter.next()
#features, labels = data
#print(features,labels)

#training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4) #batchsize = 4
print(total_samples, n_iterations)

for epochs in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader): #returns the iteration and the data
        #forward backward, update
        if(i+1)%5 == 0:
            print(f'epoch {epochs+1}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

