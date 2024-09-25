import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))


def load_model(filename):
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['layers'], model_data['biases']


def forward(X, layers, biases):
    a_values = [X]
    for i in range(len(layers)):
        z = np.dot(a_values[-1], layers[i]) + biases[i]
        a = sigmoid(z)
        a_values.append(a)
    return a_values[-1]


def predict(data_path, model_path):
    
    X_test = pd.read_csv(data_path).values

    
    layers, biases = load_model(model_path)

    
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    
    y_pred = forward(X_test, layers, biases)
    
    return y_pred

if __name__ == '__main__':
    data_path = 'train_data.csv'  
    model_path = 'saved_model.pkl'  

    
    y_true = pd.read_csv('train_labels.csv').values  

    
    y_pred = predict(data_path, model_path)
    loss = binary_cross_entropy(y_true, y_pred)

    print(f'Binary Cross-Entropy Loss: {loss:.4f}')

    
    y_pred_classes = (y_pred > 0.5).astype(int)

    
    print("Predictions: ", y_pred_classes)
