'''
Transforms like from numpy array to tensor
can be aplied to PIL images, tensors, ndarrays,or custom data

more transformation at pytorch docs transforms.html
=========On images========================
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

=========0n tensors ======================
LinearTransformation, Normalize, randomErasing


========= Conversion ====================
ToPILImage: from tensor to ndarray
ToTensor : from numpy.ndarray or PILImage


=========Generic =================
Use Lambda


=====Custom==========
write own class

=========== Compose multiple Transforms=======
composed = transforms.Compose([Rescale(256),
                                RandomCrop(224)])

torchvision.transforms.ReScale(256)
torchvision.transform.ToTensor()
'''
import torch
import torchvision
from torch.utils.data import Dataset, dataset
import numpy as np




class WineDataset(Dataset):

    def __init__(self, transform=None):
        #data loading
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        #now we dont need to transform manually
        self.x = xy[:, 1:]#torch.from_numpy(xy[:, 1:]) #inputs are all rows from column 1-end
        self.y = xy[:, [0]]#torch.from_numpy(xy[:, [0]])#classes are in the first column
        self.n_samples = xy.shape[0]
        self.transform = transform
    
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample): #makes a calable just makeing object() where object is object=ToTensor()
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))