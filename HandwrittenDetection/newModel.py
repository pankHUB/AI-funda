import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to binary format (recognize '5' or not)
y_train_binary = (y_train == 5)  # Convert training labels to binary (True for '5', False for others)
y_test_binary = (y_test == 5)    # Convert test labels to binary

# Standardize the feature values (pixel intensities)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a K-Nearest Neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed

# Train the classifier
classifier.fit(X_train, y_train_binary)

# Make predictions on the test data
y_pred_binary = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test_binary, y_pred_binary)

# Print the accuracy score
print("Accuracy:", accuracy)

# Create a confusion matrix
confusion = confusion_matrix(y_test_binary, y_pred_binary)

# Print the confusion matrix
print("Confusion Matrix:\n", confusion)

# Visualize some example predictions
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {y_pred_binary[i]}")
    plt.axis('off')
plt.show()
