# dataset: MNIST
# dataloader, Transformation
#multilayer Neural Net, activation functions
#loss and optimizer
#training loop
#model evaluation
#GPU support


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size = 784 #28x28
hidden_size = 500
num_clases = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

###############3 dataset transformation ##########################
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

#examples = iter(train_loader)
#samples, labels = examples.next()
#print(samples.shape, labels.shape)
#for i in range(6):
#    plt.subplot(2,3, i+1)
#    plt.imshow(samples[i][0], cmap='gray')
#plt.show()


################### MODEL ####################
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_clases)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

###############training loop#############################
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        #samples.shape : 100,1,28,28
        #input_size: 100, 784
        #lets reshape
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #### forward pass ######
        outputs = model(images)
        loss = criterion(outputs, labels)

        #### backward pass ######
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')



############## test and evaluation ####################
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        #value, index(class label)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0) #number of samples in the current batch
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')