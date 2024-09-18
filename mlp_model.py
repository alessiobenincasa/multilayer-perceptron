import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss Function
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

# Multilayer Perceptron
class MLP:
    def __init__(self, input_size, hidden_layers, output_size):
        self.layers = []
        self.biases = []
        self.learning_rate = 0.0314  # as given in example
        self.epochs = 84
        self.batch_size = 8
        
        # Initialize weights and biases for layers
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
                a = softmax(z)  # Softmax for the output layer
            else:
                a = sigmoid(z)
            self.a_values.append(a)
        return a
    
    def backward(self, X, y):
        # Backpropagation
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
            # Forward and backward pass for training
            for start in range(0, X_train.shape[0], self.batch_size):
                end = start + self.batch_size
                batch_X = X_train[start:end]
                batch_y = y_train[start:end]
                y_pred = self.forward(batch_X)
                self.backward(batch_X, batch_y)

            # Calculate loss for training and validation data
            y_train_pred = self.forward(X_train)
            train_loss = binary_cross_entropy(y_train, y_train_pred)

            y_valid_pred = self.forward(X_valid)
            val_loss = binary_cross_entropy(y_valid, y_valid_pred)

            # Store losses
            training_loss.append(train_loss)
            validation_loss.append(val_loss)
            
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f'Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        
        return training_loss, validation_loss

# Dataset loading and preprocessing
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Assuming 'data.csv' exists with feature columns and a target column for malignant or benign
data = pd.read_csv('data.csv')

# Split into features and target
X = data.iloc[:, :-1].values  # All feature columns
y = data.iloc[:, -1].values   # Last column for target

# Encode the target variable (M or B) into binary format
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Initialize MLP with 2 hidden layers of 24 neurons each
mlp = MLP(input_size=X_train.shape[1], hidden_layers=[24, 24], output_size=y.shape[1])

# Train the model
training_loss, validation_loss = mlp.train(X_train, y_train, X_valid, y_valid)

# Plot the learning curves
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title('Learning Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict on validation set and evaluate accuracy
y_pred = mlp.forward(X_valid)
y_pred_classes = np.argmax(y_pred, axis=1)
y_valid_classes = np.argmax(y_valid, axis=1)
accuracy = accuracy_score(y_valid_classes, y_pred_classes)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
