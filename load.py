import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data.csv')

# Display first few rows to understand the structure
print(data.head())

# Split the dataset into features (X) and target (y)
X = data.iloc[:, :-1]  # Assuming last column is the target
y = data.iloc[:, -1]   # Assuming the target column is the last one

# Split the dataset into training and validation sets (80% training, 20% validation)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split data to CSV files
X_train.to_csv('train_data.csv', index=False)
X_valid.to_csv('valid_data.csv', index=False)
y_train.to_csv('train_labels.csv', index=False)
y_valid.to_csv('valid_labels.csv', index=False)

# Visualizing the features using scatter plots
# Example: Plotting the first two features against each other
plt.figure(figsize=(10, 6))

# Plot malignant and benign data points in different colors
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='coolwarm', label='Training Data')
plt.scatter(X_valid.iloc[:, 0], X_valid.iloc[:, 1], c=y_valid, marker='x', cmap='coolwarm', label='Validation Data')

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Feature Visualization: Feature 1 vs Feature 2')
plt.legend()

# Show plot
plt.show()
