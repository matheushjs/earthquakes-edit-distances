"""forecasting_ff_nn.py

Forecasting methods using FeedForward Neural Networks

The standard here will be 1 funtion per NN architecture
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy

class MyDataset(Dataset):
    def __init__(self, distMat, targets):
        self.distMat = torch.tensor(distMat).float()
        self.targets = torch.tensor(targets).float() # Might need to change to tensor

        print(self.distMat)
        print(self.targets)
        
    def __getitem__(self, index):
        x = self.distMat[index,:]
        y = self.targets[index]
        # y = y.reshape(-1, 1)
        
        return x, y

    def __len__(self):
        return len(self.targets)

def training_procedure(
        model, train_loader, test_loader, epochs,
        earlyStoppingPatience=100, lr=0.001, log_steps=100,
        eval_steps=100
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_model = None
    best_eval_loss = float('inf')
    best_eval_corr = None
    earlyStoppingIter = 0

    train_losses = []
    eval_losses = []
    eval_corrs = []

    step = 0
    model.train()

    for epoch in range(epochs):
        if earlyStoppingIter >= earlyStoppingPatience:
            print("Early stopping triggered.")
            break

        for batch, (data, target) in enumerate(train_loader):
            model.train()
            score = model(data)
            loss = criterion(score.ravel(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # Log train loss
            train_losses.append(loss.item())

            if step % log_steps == 0:
                avgLoss = np.mean(train_losses[-log_steps:])
                print(f"Step {step} | Train Loss: {avgLoss:.6f} | Epoch: {epoch} [{batch}/{len(train_loader)}] ")

            if step % eval_steps == 0:
                # Evaluate
                model.eval()
                eval_loss = 0.0
                eval_corr = 0.0
                with torch.no_grad():
                    for test_data, test_target in test_loader:
                        test_output = model(test_data)
                        corr = np.corrcoef(test_output.detach().numpy().ravel(), test_target.detach().numpy())[0,1]
                        batch_loss = criterion(test_output.ravel(), test_target)
                        eval_loss += batch_loss.item() * len(test_data)
                        eval_corr += corr * len(test_data)
                        # print(test_output.ravel())
                        # print(test_target)
                        # print(batch_loss)
                        # print(np.mean((test_output.detach().numpy().ravel() - test_target.detach().numpy())**2))
                        # raise Exception
                        # if step == 500:
                        #     plt.scatter(test_output.ravel(), test_target, c="blue")

                    # if step == 500:
                    #     print(eval_corr / len(test_ds))
                    #     print(eval_loss / len(test_ds))
                    #     raise Exception
                eval_loss /= len(test_loader.dataset)
                eval_losses.append(eval_loss)
                eval_corr /= len(test_loader.dataset)
                eval_corrs.append(eval_corr)

                avgLoss = np.mean(train_losses[-eval_steps:])
                print(f"Step {step} | Eval Loss: {eval_loss:.6f} | Corr: {eval_corr:.6f} | Epoch: {epoch} [{batch}/{len(train_loader)}] ")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model = copy.deepcopy(model)
                    best_eval_corr = eval_corr
                    earlyStoppingIter = 0
                else:
                    earlyStoppingIter += 1

                if earlyStoppingIter >= earlyStoppingPatience:
                    print("Early stopping.")
                    break

    return best_model, best_eval_loss, best_eval_corr, train_losses, eval_losses, eval_corrs

def predict_ff_nn(
        y, trainSize, distMat=None, seisFeatures=None, numBases=100,
        batch_size=128, log_steps=100, eval_steps=100, si_activation="relu",
        plot=False
):
    if distMat is None and seisFeatures is None:
        raise Exception("'distMat' and 'seisFeatures' cannot be both None.")

    yNormalizer = np.mean(y)
    y = np.array(y) / yNormalizer

    trainX = []
    testX  = []

    if distMat is not None:
        distMat = distMat / np.mean(distMat) / 2

        #np.random.seed(11)

        idx = np.arange(trainSize)
        np.random.shuffle(idx)
        idx = idx[:numBases]
        
        trainX.append( distMat[:trainSize,idx] )
        testX.append(  distMat[trainSize:,idx] )
    else:
        raise Exception("Not implemented.")

    trainY = y[:trainSize]
    testY  = y[trainSize:]

    train_ds = MyDataset(trainX, trainY)
    test_ds  = MyDataset(testX, testY)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = neural_network(trainX)

    epochs = 10000
    lr = 0.001

    best_model, best_eval_loss, best_eval_corr, \
        train_losses, eval_losses, eval_corrs = \
            training_procedure(model, train_loader, test_loader,
                               epochs=epochs, lr=lr)

    trainY = trainY * yNormalizer
    testY = testY * yNormalizer

    # Plotting
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
        eval_x = np.arange(eval_steps, eval_steps * len(eval_losses) + 1, eval_steps)
        plt.plot(eval_x, eval_losses, label='Eval Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training vs Evaluation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    with torch.no_grad():
        x, y = train_ds[:]
        best_model.eval()
        predicted = best_model(x).detach().numpy().ravel()*yNormalizer
        if plot:
            plt.figure(figsize=(10, 5))
            plt.scatter(trainY, predicted)
            plt.show()
        print(np.corrcoef(trainY, predicted)[0,1])
        print(np.mean((trainY - predicted)**2))

        x, y = test_ds[:]
        best_model.eval()
        predicted = best_model(x).detach().numpy().ravel()*yNormalizer
        if plot:
            plt.figure(figsize=(10, 5))
            plt.scatter(testY, predicted)
            plt.show()
        corr = np.corrcoef(testY, predicted)[0,1]
        print(corr)
        mse = np.mean((testY - predicted)**2)
        print(mse)

    return predicted, testY, corr, mse