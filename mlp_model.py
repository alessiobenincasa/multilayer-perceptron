import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import argparse


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))


class MLP:
    def __init__(self, input_size, hidden_layers, output_size):
        self.layers = []
        self.biases = []
        self.learning_rate = 0.0314  
        self.epochs = 84
        self.batch_size = 8
        
        
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def forward(self, X):
        self.z_values = []
        self.a_values = [X]
        for i in range(len(self.layers)):
            z = np.dot(self.a_values[-1], self.layers[i]) + self.biases[i]
            self.z_values.append(z)
            if i == len(self.layers) - 1:
                a = softmax(z)  
            else:
                a = sigmoid(z)
            self.a_values.append(a)
        return a
    
    def backward(self, X, y):
        
        m = y.shape[0]
        dz = self.a_values[-1] - y
        for i in reversed(range(len(self.layers))):
            dw = np.dot(self.a_values[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            if i > 0:
                dz = np.dot(dz, self.layers[i].T) * sigmoid_derivative(self.a_values[i])
            self.layers[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def train(self, X_train, y_train, X_valid, y_valid):
        training_loss = []
        validation_loss = []
        for epoch in range(self.epochs):
            
            for start in range(0, X_train.shape[0], self.batch_size):
                end = start + self.batch_size
                batch_X = X_train[start:end]
                batch_y = y_train[start:end]
                y_pred = self.forward(batch_X)
                self.backward(batch_X, batch_y)

            
            y_train_pred = self.forward(X_train)
            train_loss = binary_cross_entropy(y_train, y_train_pred)

            y_valid_pred = self.forward(X_valid)
            val_loss = binary_cross_entropy(y_valid, y_valid_pred)

            
            training_loss.append(train_loss)
            validation_loss.append(val_loss)
            
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f'Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        
        return training_loss, validation_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Multilayer Perceptron on breast cancer dataset.")
    parser.add_argument('--layers', nargs='+', type=int, default=[24, 24], help='Number of neurons in hidden layers.')
    parser.add_argument('--epochs', type=int, default=84, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=0.0314, help='Learning rate for gradient descent.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--data', type=str, default='data.csv', help='Path to the dataset CSV file.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    
    data = pd.read_csv(args.data)

    
    print(data.dtypes)  

    
    data = data.select_dtypes(include=[np.number])  

    
    y = data.iloc[:, -1].values   
    X = data.iloc[:, :-1].values  

    
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  
    X_valid = scaler.transform(X_valid)      

    
    mlp = MLP(input_size=X_train.shape[1], hidden_layers=args.layers, output_size=y.shape[1])
    mlp.learning_rate = args.learning_rate
    mlp.epochs = args.epochs
    mlp.batch_size = args.batch_size

    
    training_loss, validation_loss = mlp.train(X_train, y_train, X_valid, y_valid)

    
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    y_pred = mlp.forward(X_valid)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_valid_classes = np.argmax(y_valid, axis=1)
    accuracy = accuracy_score(y_valid_classes, y_pred_classes)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')
