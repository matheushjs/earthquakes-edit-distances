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

batch_size = 128
lr = 0.001
epochs = 50000
earlyStoppingPatience = 100

def training_procedure(
        model, train_loader, test_loader, epochs,
        earlyStoppingPatience=100, lr=0.001, log_steps=100,
        eval_steps=100):
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

def predict_ff_nn(y, distMat, trainSize, eps, si_activation="relu"):
    pass