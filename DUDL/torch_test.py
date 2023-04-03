import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

N = 30
x = torch.randn(N,1)/2
y = x + torch.randn(N,1)/2

#making the model
ANNreg = nn.Sequential(
    nn.Linear(1,1),
    nn.ReLU(),
    nn.Linear(1,1)
    )

ANNreg

# learning rate
learningRate = 0.05

#loss function
lossfun = nn.MSELoss()

#optimizer
optimizer = torch.optim.SGD(ANNreg.parameters(),lr=learningRate)

#Training the model
numepochs = 500
losses = torch.zeros(numepochs)

for empoch in range(numepochs):
    yHat =  ANNreg(x)

#comptuting the loss
loss = lossfun(yHat,y)
losses[empoch] = loss

#back porp
optimizer.zero_grad()
loss.backward()
optimizer.step()

predictions = ANNreg(x)

testloss = (predictions-y).pow(2).mean()

plt.figure(1)
plt.plot(numepochs,testloss.detach(),'ro')
plt.show()

plt.plot(x,y,'s')
plt.show()

