import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv('data.csv')


print(data.head())


y = data.iloc[:, 1]  


X = data.drop(data.columns[1], axis=1)  


y = y.map({'M': 1, 'B': 0})


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.to_csv('train_data.csv', index=False)
X_valid.to_csv('valid_data.csv', index=False)
y_train.to_csv('train_labels.csv', index=False)
y_valid.to_csv('valid_labels.csv', index=False)


plt.figure(figsize=(10, 6))


plt.scatter(X_train.iloc[:, 1], X_train.iloc[:, 2], c=y_train, cmap='coolwarm', label='Training Data')
plt.scatter(X_valid.iloc[:, 1], X_valid.iloc[:, 2], c=y_valid, marker='x', cmap='coolwarm', label='Validation Data')


plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Feature Visualization: Feature 1 vs Feature 2')
plt.legend()


plt.show()
