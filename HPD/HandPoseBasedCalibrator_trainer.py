''' hand pose based depth estimation trainer'''

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from models import TransformerModel

filePath = 'HandDepth'
fileName = 'XY'
X = []
Y = []
Z = []
with open(f'./{filePath}/{fileName}.txt', 'r') as f:
    line = f.readline()
    while line:
        line = line.strip().split(',')
        tmp1 = []
        tmp2 = []
        for i in range(21):
            x, y = int(line[2*i]), int(line[2*i+1])
            tmp1.append(x)
            tmp2.append(y)
        X.append(tmp1)
        Y.append(tmp2)
        line = f.readline()


fileName = 'Depth'
with open(f'./{filePath}/{fileName}.txt', 'r') as f:
    line = f.readline()
    while line:
        line = line.strip().split(',')
        tmp = []
        for i in range(21):
            z = int(line[i])
            tmp.append(z)
        Z.append(tmp)
        line = f.readline()
size = len(X)

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)
Z = scaler.fit_transform(Z)

X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()
Z = torch.from_numpy(Z).float()

model = TransformerModel()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(500):
    optimizer.zero_grad()
    output = model(X, Y)
    loss = criterion(output, Z)
    loss.backward()
    optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 50, loss.item()))

predictions = model(X, Y)

mse = torch.mean(torch.square(predictions - Z)).item()
mae = torch.mean(torch.abs(predictions - Z)).item()

print("MSE:", mse)
print("MAE:", mae)

torch.save(model.state_dict(), 'transformer_model.pth')