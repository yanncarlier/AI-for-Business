#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Assignment 1: Linear Regression on Real Estate Prices
Predicting House Price per Unit Area using Multiple Features
Due: Feb 22, 9:00 AM | 100 points
Dataset: RealEstate.csv (Taipei housing data)
"""
# ===========================
# 1. IMPORT LIBRARIES
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Configure plot appearance (for clean, readable output)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
# ===========================
# 2. LOAD THE DATASET
# ===========================
# Load the real estate dataset
# Note: A DtypeWarning may appear — safe to ignore for this dataset
data = pd.read_csv('RealEstate.csv')
# Display first few rows for verification
print("Dataset preview:")
print(data.head())
print("\n")
# ===========================
# 3. INITIAL DATA VISUALIZATION
# ===========================
# Scatter plot: House Age vs Price per Unit Area (expect weak linear relationship)
plt.scatter(data['House age'], data['House price of unit area'], alpha=0.6)
plt.title('House Age vs House Price per Unit Area')
plt.xlabel('House Age (years)')
plt.ylabel('House Price per Unit Area')
plt.grid(True, alpha=0.3)
plt.savefig('scatter_age_vs_price.png', dpi=300, bbox_inches='tight')
plt.close()
print("Scatter plot saved as 'scatter_age_vs_price.png'\n")
# ===========================
# 4. SIMPLE 1D REGRESSION: House Age Only
# ===========================
# Select features for the simple model
selected_data = data[['House age', 'House price of unit area']]
# Separate features (X) and target (y)
X_simple = selected_data.drop('House price of unit area', axis=1)
y = selected_data['House price of unit area']
# Train-test split (80% train, 20% test) — fixed random state for reproducibility
x_train, x_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)
print(f"Training set: {x_train.shape[0]} samples")
print(f"Test set:     {x_test.shape[0]} samples")
print(f"Test split:   {100 * x_test.shape[0] / len(data):.2f}%")
print("\n")
# ===========================
# 5. MODEL BUILDING FUNCTION
# ===========================


def generate_model(X_train, y_train):
    """
    Trains a LinearRegression model and returns the fitted model.
    Parameters:
        X_train: Training features
        y_train: Training target values
    Returns:
        fitted sklearn LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Train the simple 1D model
simple_model = generate_model(x_train, y_train)
# Display the learned equation (for 1 feature)
coef = simple_model.coef_[0]
intercept = simple_model.intercept_
print(f"Simple Model Equation:")
print(f"Price = {coef:.6f} × (House Age) + {intercept:.6f}\n")
# ===========================
# 6. PREDICTIONS & EVALUATION (1D Model)
# ===========================
# Predict on test set
predicted_values = simple_model.predict(x_test)
# Calculate Mean Squared Error
mse_1d = metrics.mean_squared_error(y_test, predicted_values)
print(f"1D Model (House Age only) — Mean Squared Error: {mse_1d:.2f}\n")
# Visualization: True vs Predicted (with regression line)
plt.scatter(x_test, y_test, color='gray', label='Actual', alpha=0.7)
plt.plot(x_test, predicted_values, color='red', linewidth=2, label='Predicted')
plt.title('1D Linear Regression: House Age → Price per Unit Area')
plt.xlabel('House Age')
plt.ylabel('House Price per Unit Area')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('regression_1d_fit.png', dpi=300, bbox_inches='tight')
plt.close()
print("1D regression plot saved as 'regression_1d_fit.png'\n")
# ===========================
# 7. HIGHER-DIMENSIONAL REGRESSION (Multiple Features)
# ===========================
# Select meaningful predictive features (excluding ID and target)
hd_features = [
    'House age',
    'Distance to the nearest MRT station',
    'Number of convenience stores',
    'Latitude',
    'Longitude'
]
# Reconstruct full-feature training/test sets using original indices
hd_x_train = data.loc[x_train.index, hd_features]
hd_x_test = data.loc[x_test.index,  hd_features]
# Train multi-feature model
hd_model = LinearRegression()
hd_model.fit(hd_x_train, y_train)
# Predictions and error
hd_predictions = hd_model.predict(hd_x_test)
hd_mse = metrics.mean_squared_error(y_test, hd_predictions)
print("Higher-Dimensional Model Features:")
for feature in hd_features:
    print(f"   • {feature}")
print(f"\nHigher-Dimensional Model — Mean Squared Error: {hd_mse:.2f}")
print(
    f"Improvement over 1D model: {((mse_1d - hd_mse) / mse_1d * 100):.1f}% reduction in error!\n")
# ===========================
# 8. FINAL VISUALIZATION: Actual vs Predicted (Top 25)
# ===========================
# Compare actual vs predicted values side-by-side
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': hd_predictions
}).head(25)
comparison_df.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('Actual vs Predicted House Price per Unit Area (First 25 Test Samples)')
plt.ylabel('Price per Unit Area')
plt.xlabel('Test Sample Index')
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('actual_vs_predicted_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("Bar chart saved as 'actual_vs_predicted_bar.png'\n")
print("=== Assignment Complete ===")
print("All plots have been saved successfully.")
print("You achieved excellent results with multi-feature regression!")
