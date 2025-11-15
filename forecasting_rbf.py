import numpy as np

def predict_rbf(y, distMat, trainSize, eps, pseudoinv=False):
    # Find the weights using only the training set
    trainMat = distMat[:trainSize,:]
    
    trainGram = np.exp(-trainMat**2 / eps)

    # Add bias parameters 
    trainGram = np.hstack([ np.ones((trainGram.shape[0], 1)), trainGram ])

    # Calculate weights
    try:
        if pseudoinv:
            w = np.linalg.pinv(trainGram.transpose() @ trainGram) @ (trainGram.transpose() @ y[:trainSize])
        else:
            w = np.linalg.inv(trainGram.transpose() @ trainGram) @ (trainGram.transpose() @ y[:trainSize])
    except:
        real = y[trainSize:]
        return np.zeros(len(real)), real

    # Proceed to testing
    testMat = distMat[trainSize:,:]

    testGram = np.exp(-testMat**2 / eps)

    # To plot results on train data
    #plt.scatter(trainGram @ w, y[:trainSize])
    
    # Add bias parameters
    testGram = np.hstack([ np.ones([testGram.shape[0], 1]), testGram ])

    predicted = testGram @ w
    
    return predicted, y[trainSize:]