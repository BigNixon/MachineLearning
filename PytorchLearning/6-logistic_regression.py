import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#0) prepare data
bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)
    #scale
sc = StandardScaler() #0 mean and varianze
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# converting into tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

#reshaping into columns
Y_train = Y_train.view(Y_train.shape[0], 1)
Y_test = Y_test.view(Y_test.shape[0], 1)


# 1) model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)


# 2) loss and optimizer
learning_rate=0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#3) training loop
num_epochs = 1000
for epochs in range(num_epochs):
    #fordward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, Y_train)

    #backwad pass dloss/dw
    loss.backward()

    #updates
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    if(epochs+1)%10 ==0:
        print(f'epoch: {epochs+1}, loss={loss.item():.3f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f'accuracy= {acc:.4f}')









