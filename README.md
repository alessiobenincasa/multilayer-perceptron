# multilayer-perceptron
Multilayer Perceptron Project
Overview
This project is an introduction to artificial neural networks through the implementation of a Multilayer Perceptron (MLP) to classify breast cancer as either malignant or benign, based on the Wisconsin Breast Cancer dataset.

Project Structure
The project requires you to build and train a neural network from scratch, without the use of external libraries that handle neural networks. The implementation includes:

Data Processing:

The dataset is provided and contains features that describe the characteristics of cell nuclei from breast mass tissue samples.
Preprocess the data before training and split it into training and validation sets.
Neural Network Structure:

You must implement a feedforward neural network with at least two hidden layers.
Each layer should consist of perceptrons with customizable activation functions.
The output layer will use the softmax function for binary classification (malignant vs. benign).
Training:

Use backpropagation and gradient descent to train the model.
Display training and validation metrics at each epoch to monitor progress.
Evaluation:

Evaluate the model using binary cross-entropy loss.
Implement learning curves to visualize the loss and accuracy during training.
Mandatory Programs
You will create three main programs:

Data Splitting Program:

This program will split the dataset into training and validation sets.
To run the program:
bash
Copier le code
python load.py
Training Program:

Train the MLP using backpropagation and gradient descent.
Save the model (network topology and weights) after training.
To run the program:
bash
Copier le code
python train.py
Prediction Program:

Load the saved model, perform predictions, and evaluate its accuracy.
To run the program:
bash
Copier le code
python predict.py
