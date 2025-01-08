import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sigmoid function with clipping to avoid overflow
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow in exp
    return 1 / (1 + np.exp(-z))

# Gradient Descent for Logistic Regression
def gradient_desc(X, Y, m, alpha, iterations, theta):
    for _ in range(iterations):
        h_theta = sigmoid(X.dot(theta))  # Compute the hypothesis
        gradient = (1 / m) * X.T.dot(h_theta - Y)  # Gradient calculation
        theta = theta - alpha * gradient  # Update theta
    return theta

# Load the dataset
classification_data = pd.read_csv('Dataset.csv.csv')
print(classification_data)

# Extract features and target
Y = classification_data['Outcome'].values.reshape(-1, 1)  # Reshape Y to column vector
x1 = classification_data['Age'].values
x2 = classification_data['BMI'].values
x3 = classification_data['BloodPressure'].values
x4 = classification_data['GlucoseLevel'].values
x5 = classification_data['InsulinLevel'].values
x6 = classification_data['PhysicalActivity'].values

# Add bias term (x0 = 1)
x0 = np.ones(len(x1))
X = np.array([x0, x1, x2, x3, x4, x5, x6]).T

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Initialize parameters
theta = np.zeros((7, 1))  # Initialize theta

alpha = 0.05  # Learning rate
iterations = 1000  # Number of iterations
m = len(Y_train)  # Number of training examples

# Perform Gradient Descent
final_theta = gradient_desc(X_train, Y_train, m, alpha, iterations, theta)
#print(final_theta)

# Predictions on the test set
temp = X_test.dot(final_theta)
h_theta = sigmoid(temp)
predictions = (h_theta >= 0.5).astype(int)

# Calculate accuracy and confusion matrix
a_s = accuracy_score(Y_test, predictions)
print("Accuracy Score:", a_s)

c_m = confusion_matrix(Y_test, predictions)
print("Confusion Matrix:\n", c_m)


