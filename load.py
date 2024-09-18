import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data.csv')

# Display the first few rows of the data to ensure it's loaded correctly
print(data.head())

# "Diagnosis" is the second column (index 1). Extract it as the target labels (y)
y = data.iloc[:, 1]  # Second column (diagnosis)

# Drop the second column ("diagnosis") to get the features (X)
X = data.drop(data.columns[1], axis=1)  # All columns except the second column

# Map the 'M' and 'B' labels to 1 and 0 respectively
y = y.map({'M': 1, 'B': 0})

# Split the data into training and validation sets (80% training, 20% validation)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split datasets to CSV files
X_train.to_csv('train_data.csv', index=False)
X_valid.to_csv('valid_data.csv', index=False)
y_train.to_csv('train_labels.csv', index=False)
y_valid.to_csv('valid_labels.csv', index=False)

# Plotting the features
plt.figure(figsize=(10, 6))

# Visualize the training and validation data (scatter plot using the first two features)
plt.scatter(X_train.iloc[:, 1], X_train.iloc[:, 2], c=y_train, cmap='coolwarm', label='Training Data')
plt.scatter(X_valid.iloc[:, 1], X_valid.iloc[:, 2], c=y_valid, marker='x', cmap='coolwarm', label='Validation Data')

# Add labels and title to the plot
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Feature Visualization: Feature 1 vs Feature 2')
plt.legend()

# Show the plot
plt.show()
