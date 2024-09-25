import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))


def calculate_accuracy(y_true, y_pred):
    y_pred_classes = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred_classes)

class MLP:
    def __init__(self, input_size, hidden_layers, output_size=1):
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
        self.a_values = [X]
        for i in range(len(self.layers)):
            z = np.dot(self.a_values[-1], self.layers[i]) + self.biases[i]
            a = sigmoid(z)
            self.a_values.append(a)
        return self.a_values[-1]

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
        training_accuracy = []
        validation_accuracy = []

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

            train_acc = calculate_accuracy(y_train, y_train_pred)
            val_acc = calculate_accuracy(y_valid, y_valid_pred)

            training_loss.append(train_loss)
            validation_loss.append(val_loss)
            training_accuracy.append(train_acc)
            validation_accuracy.append(val_acc)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f'Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - '
                      f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        return training_loss, validation_loss, training_accuracy, validation_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Multilayer Perceptron on a binary classification dataset.")
    parser.add_argument('--train', type=str, required=True, help='Path to the training dataset CSV file')
    parser.add_argument('--valid', type=str, required=True, help='Path to the validation dataset CSV file')
    parser.add_argument('--layers', nargs='+', type=int, default=[24, 24], help='Number of neurons in hidden layers.')
    parser.add_argument('--epochs', type=int, default=84, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=0.0314, help='Learning rate for gradient descent.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    
    X_train = pd.read_csv(args.train).values
    X_valid = pd.read_csv(args.valid).values
    y_train = pd.read_csv('train_labels.csv').values
    y_valid = pd.read_csv('valid_labels.csv').values

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    
    mlp = MLP(input_size=X_train.shape[1], hidden_layers=args.layers, output_size=1)
    mlp.learning_rate = args.learning_rate
    mlp.epochs = args.epochs
    mlp.batch_size = args.batch_size

    training_loss, validation_loss, training_accuracy, validation_accuracy = mlp.train(X_train, y_train, X_valid, y_valid)

    
    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Learning Curves - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    
    plt.figure(figsize=(10, 5))
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.title('Learning Curves - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    
    model_data = {'layers': mlp.layers, 'biases': mlp.biases}
    with open('saved_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print('Model saved as saved_model.pkl')
