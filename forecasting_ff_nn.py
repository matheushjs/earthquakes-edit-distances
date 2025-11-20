"""forecasting_ff_nn.py

Forecasting methods using FeedForward Neural Networks

The standard here will be 1 funtion per NN architecture
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from data_loaders import EQTimeWindows, load_dataset
import numpy as np
import matplotlib.pyplot as plt
import copy

# \ell norm
def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)

# Gaussian RBF
def rbf_gaussian(x, sigma1=1, sigma2=1):
    return sigma1*torch.exp(-torch.pow(x / sigma2,2))

class MyDataset(Dataset):
    def __init__(self, targets, distMat=None, seisFeatures=None):
        self.distMat = torch.tensor(distMat).float() if distMat is not None else None
        self.seisFeatures = [ torch.tensor(i).float() for i in seisFeatures ] if seisFeatures is not None else None
        self.targets = torch.tensor(targets).float() # Might need to change to tensor

        # print(self.distMat)
        # print(self.targets)

    def __getitem__(self, index):
        x1 = self.distMat[index,:] if self.distMat is not None else torch.tensor(float("NaN"))
        x2 = [ feats[index,:] for feats in self.seisFeatures ] if self.seisFeatures is not None else torch.tensor(float("NaN"))
        y = self.targets[index]
        # y = y.reshape(-1, 1)
        
        return {
                "distMat": x1,
                "seisFeatures": x2
            }, y

    def __len__(self):
        return len(self.targets)

def get_predictions(loader, model):
    """Helper function to run inference using a DataLoader."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        # Your loader yields (data, target) tuples
        for data, target in loader:
            preds = model(data).detach().numpy().ravel()
            all_preds.extend(preds)
            all_targets.extend(target.detach().numpy().ravel())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    return all_preds, all_targets

def training_procedure(
        model, train_loader, test_loader, epochs,
        earlyStoppingPatience=100, lr=0.001, log_steps=100,
        eval_steps=100, verbose=False
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
            if verbose:
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

            if verbose and step % log_steps == 0:
                avgLoss = np.mean(train_losses[-log_steps:])
                print(f"Step {step} | Train Loss: {avgLoss:.6f} | Epoch: {epoch} [{batch}/{len(train_loader)}] ")

            if step % eval_steps == 0:
                # Evaluate
                model.eval()

                predicted, real = get_predictions(test_loader, model)
                eval_loss = criterion(torch.tensor(predicted), torch.tensor(real))
                eval_losses.append(eval_loss)
                eval_corr = np.corrcoef(predicted, real)[0,1]
                eval_corrs.append(eval_corr)

                avgLoss = np.mean(train_losses[-eval_steps:])
                if verbose:
                    print(f"Step {step} | Eval Loss: {eval_loss:.6f} | Corr: {eval_corr:.6f} | Epoch: {epoch} [{batch}/{len(train_loader)}] ")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model = copy.deepcopy(model)
                    best_eval_corr = eval_corr
                    earlyStoppingIter = 0
                else:
                    earlyStoppingIter += 1

                if earlyStoppingIter >= earlyStoppingPatience:
                    if verbose:
                        print("Early stopping.")
                    break

    return best_model, best_eval_loss, best_eval_corr, train_losses, eval_losses, eval_corrs

def predict_ff_nn(
        y, trainSize, distMat=None, seisFeatures=None, numBases=100,
        batch_size=128, log_steps=100, eval_steps=100, si_activation="relu",
        earlyStoppingPatience=100, lr=0.001, plot=False, verbose=False
):
    if distMat is None and seisFeatures is None:
        raise Exception("'distMat' and 'seisFeatures' cannot be both None.")

    y = np.array(y)
    trainY_raw = y[:trainSize]
    testY_raw  = y[trainSize:]

    y_mean = np.mean(trainY_raw)
    y_std  = np.std(trainY_raw)

    trainY = (trainY_raw - y_mean) / y_std
    testY  = (testY_raw - y_mean) / y_std

    class neural_network(nn.Module):
        # Hierarchical neural network
        # First layer is composed of subnetworks
        def __init__(self):
            super(neural_network, self).__init__()

            self.numFeatures = numBases

            self.log_sigma1 = torch.nn.Parameter(torch.zeros(self.numFeatures))
            self.log_sigma2 = torch.nn.Parameter(torch.zeros(self.numFeatures))

            self.ed_fc1 = nn.Linear(in_features=self.numFeatures, out_features=1)

            self.si_fc1 = [
                nn.Linear(in_features=featWindow.shape[1], out_features=1)
                for featWindow in seisFeatures
            ]

            featureCount = self.ed_fc1.out_features + sum([ i.out_features for i in self.si_fc1])
            self.out_fc = nn.Linear(in_features=featureCount, out_features=1)

            self.bn = nn.BatchNorm1d(1)

        # x is whatever you set the __getitem__ of the Dataset object to be.
        def forward(self, x):
            layers = []

            if distMat is not None:
                ed_x = rbf_gaussian(x["distMat"],
                                torch.exp(self.log_sigma1),
                                torch.exp(self.log_sigma2))
                ed_x = self.ed_fc1(ed_x)
                layers.append(ed_x)

            if seisFeatures is not None:
                for l, f in zip(self.si_fc1, x["seisFeatures"]):
                    layers.append(l(f))

            x = torch.concatenate(layers, dim=1)
            x = self.out_fc(x)

            if x.shape[0] > 1:
                x = self.bn(x)

            return x

    if distMat is not None:
        distMat = distMat / np.mean(distMat) / 2

        #np.random.seed(11)

        idx = np.arange(trainSize)
        np.random.shuffle(idx)
        idx = idx[:numBases]
        
        ed_trainX = distMat[:trainSize,idx]
        ed_testX  = distMat[trainSize:,idx]
    
    if seisFeatures is not None:
        seisFeatures_train = [ i[:trainSize] for i in seisFeatures ]
        seisFeatures_test = [ i[trainSize:] for i in seisFeatures ]

    train_ds = MyDataset(trainY, distMat=ed_trainX, seisFeatures=seisFeatures_train)
    test_ds  = MyDataset(testY, distMat=ed_testX, seisFeatures=seisFeatures_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = neural_network()

    epochs = 10000

    best_model, best_eval_loss, best_eval_corr, \
        train_losses, eval_losses, eval_corrs = \
            training_procedure(model, train_loader, test_loader, epochs=epochs,
                               lr=lr, earlyStoppingPatience=earlyStoppingPatience,
                               log_steps=log_steps, eval_steps=eval_steps, verbose=verbose)

    trainY = trainY * y_std + y_mean
    testY = testY * y_std + y_mean

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

    predicted_train, real_train = get_predictions(train_loader, best_model)
    predicted_train = predicted_train * y_std + y_mean
    real_train = real_train * y_std + y_mean

    if plot:
        plt.figure(figsize=(10, 5))
        plt.scatter(real_train, predicted_train)
        plt.show()
    if verbose:
        print(np.corrcoef(real_train, predicted_train)[0,1])
        print(np.mean((real_train - predicted_train)**2))

    predicted, _ = get_predictions(test_loader, best_model)
    predicted = predicted * y_std + y_mean

    if plot:
        plt.figure(figsize=(10, 5))
        plt.scatter(testY, predicted)
        plt.show()
    corr = np.corrcoef(testY, predicted)[0,1]
    mse = np.mean((testY - predicted)**2)
    
    if verbose:
        print(corr)
        print(mse)

    return predicted, testY, corr, mse

if __name__ == "__main__":
    data = load_dataset("ja")
    distMat = np.load("/media/mathjs/HD-ADU3/distance-matrix-ja-minmag0-inputw7-outputw1-tlambda100.npy")

    eqtw = EQTimeWindows(data, 7, 1, nthreads = 22)

    trainSize = distMat.shape[0]//2
    trainMat = distMat[:trainSize,:trainSize]
    eps = 2 * np.mean(trainMat[trainMat > 0])**2

    logN = [ np.log(len(i) + 1) for i in eqtw.y_quakes[0] ]
    mmags = eqtw.getYQuakesMaxMag()[0]

    a = predict_ff_nn(logN[1:], trainSize, distMat=distMat,
                      earlyStoppingPatience=50, lr=0.01, plot=True,
                      log_steps=1, eval_steps=10, batch_size=128)

    print(a)