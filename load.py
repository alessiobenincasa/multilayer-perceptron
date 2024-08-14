import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Dataset
# Replace 'path_to_your_dataset.csv' with the actual path to your dataset
df = pd.read_csv('data.csv')

# Step 2: Basic Exploration

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Get summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check the data types of each column
print("\nData types of each column:")
print(df.info())

# Step 3: Visualize the Data

# Histograms for each feature
print("\nVisualizing the distribution of each feature with histograms...")
df.hist(bins=30, figsize=(20, 15), color='skyblue')
plt.suptitle("Histogram of All Features", fontsize=16)
plt.show()

# Correlation matrix and heatmap
print("\nVisualizing the correlation matrix with a heatmap...")
plt.figure(figsize=(16, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.show()

# Pairplot for a few selected features (optional)
selected_features = ['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'diagnosis']
sns.pairplot(df[selected_features], hue='diagnosis', palette='Set1')
plt.suptitle("Pairplot of Selected Features", y=1.02, fontsize=16)
plt.show()
