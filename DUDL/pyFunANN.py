# import libraries
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

# create data

def createData(N, m):
    '''
    Function that creates random 
    data given a sample size.
    Returns two lists of length N
    '''
    x = torch.randn(N,1)
    y = m*x + torch.randn(N,1)/2
    return(x,y)

#plot data using function
x1, y1 = createData(50, 1)
plt.plot(x1,y1,'s')
# and plot
#plt.plot(x,y,'s')
# create data

def ANNRunner(x, y):
    '''
    Function to train a 
    model on some data
    '''

    # build model
    ANNreg = nn.Sequential(
        nn.Linear(1,1),  # input layer
        nn.ReLU(),       # activation function
        nn.Linear(1,1)   # output layer
        )

    ANNreg

    '''
    # and plot
    plt.plot(x1,y1,'s')
    plt.show()# learning rate
    
    '''
    learningRate = .05
    # loss function
    lossfun = nn.MSELoss()

    # optimizer (the flavor of gradient descent to implement)
    optimizer = torch.optim.SGD(ANNreg.parameters(),lr=learningRate)# train the model
    numepochs = 500
    losses = torch.zeros(numepochs)


    ## Train the model!
    for epochi in range(numepochs):

        # forward pass
        yHat = ANNreg(x)

        # compute loss
        loss = lossfun(yHat,y)
        losses[epochi] = loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()# show the losses

    # manually compute losses
    # final forward pass
    predictions = ANNreg(x)

    # final loss (MSE)
    testloss = (predictions-y).pow(2).mean()

    return (predictions, testloss)


#Set up empty list
grads = []
losses = []
TestRuns = 50

# Set up for loop of gradients
for grad in np.linspace(-2,2,21):
    runloss  = []
    for run in range(TestRuns): 
        # Create data for run
        x,y = createData(50, grad)
        #train model on data
        preds, loss = ANNRunner(x,y)
        runloss.append(loss.detach())
    avloss = sum(runloss)/TestRuns
    grads.append(grad)
    losses.append(avloss)

plt.plot(grads,losses)
plt.xlabel('Gradients')
plt.ylabel('Losses')
plt.title('Gradients Vs Losses')
plt.show()


'''
plt.plot(losses.detach(),'o',markerfacecolor='w',linewidth=.1)
plt.plot(numepochs,testloss.detach(),'ro')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Final loss = %g' %testloss.item())
plt.show()
testloss.item()
# plot the data
plt.plot(x1,y1,'bo',label='Real data')
plt.plot(x1,predictions.detach(),'rs',label='Predictions')
plt.title(f'prediction-data r={np.corrcoef(y.T,predictions.detach().T)[0,1]:.2f}')
plt.legend()
plt.show()'''