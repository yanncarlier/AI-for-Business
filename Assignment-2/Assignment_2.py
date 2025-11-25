#!/usr/bin/env python3
# Assignment_2.py
# Handwritten digit classification with scikit-learn Logistic Regression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# -------------------------------------------------
# Load the digits dataset
# -------------------------------------------------
digits = load_digits()
# -------------------------------------------------
# Q1: Volume of the dataset (number of samples)
# -------------------------------------------------
volume = len(digits.data)
print(f"1. Volume of the dataset: {volume}")          # → 1797
# -------------------------------------------------
# Q2: Dimensionality of each image (8×8 = 64 pixels)
# -------------------------------------------------
dimensionality = digits.data.shape[1]
print(f"2. Dimensionality of each image: {dimensionality}")   # → 64
# -------------------------------------------------
# Q3: First 5 labels
# -------------------------------------------------
labels = digits.target[:5]
print(f"3. First 5 labels: {labels}")                  # → [0 1 2 3 4]
# -------------------------------------------------
# Q4: Train-test split (80% train, 20% test)
# -------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    digits.data,
    digits.target,
    test_size=0.2,
    random_state=42
)
# -------------------------------------------------
# Q5 & Q6: Create and train Logistic Regression model
# -------------------------------------------------
# increased iterations for convergence
logisticReg = LogisticRegression(max_iter=10000)
logisticReg.fit(X_train, Y_train)
print("Model training completed.")
# -------------------------------------------------
# Q7: Evaluate the model on the test set
# -------------------------------------------------
score = logisticReg.score(X_test, Y_test)
print(f"Test accuracy: {score:.4f}  ({score*100:.2f}%)")   # usually ~96-98%
# -------------------------------------------------
# Optional: Show a quick prediction example
# -------------------------------------------------
print("\nExample prediction on the first test image:")
pred = logisticReg.predict(X_test[0].reshape(1, -1))[0]
true = Y_test[0]
print(f"Predicted digit: {pred} | True digit: {true}")
