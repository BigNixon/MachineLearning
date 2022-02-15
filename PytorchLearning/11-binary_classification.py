import torch
import torch.nn as nn

class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # y[is, hs] = W[]*X[is] + B
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1) #y[hs,1] = W[]*X[] +B
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear1(out)
        #sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred


model = NeuralNet1(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.BCELoss()