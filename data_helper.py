import numpy as np

def get_data(folder='data/'):
    X = np.load(folder+'X_all.npy')
    y = np.load(folder+'y_all.npy').reshape(-1)
    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    X_train = X[:len(X)//2]
    y_train = y[:len(y)//2]
    X_val = X[len(X)//2:]
    y_val = y[len(y)//2:]
    return X_train, y_train, X_val, y_val