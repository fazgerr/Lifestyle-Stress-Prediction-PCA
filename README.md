# 🧠 Lifestyle-Based Stress Level Prediction using PCA

## 📌 Project Overview
This project predicts individual stress levels (on a scale of 1-10) based on daily lifestyle factors using dimensionality reduction and machine learning regression techniques.

## 📊 Dataset & Features
Analyzed the "Sleep Health and Lifestyle" dataset containing behavioral and physiological metrics:
* Sleep Duration & Quality of Sleep
* Physical Activity Level & Daily Steps
* Resting Heart Rate

## ⚙️ Methodology
1. **Data Preprocessing:** Applied Z-score standardization to ensure variables on different scales contribute equally.
2. **Dimensionality Reduction (PCA):** Addressed multicollinearity by reducing the 5-dimensional feature space into 2 interpretable Principal Components:
   * *PC1 (Sleep-Recovery Axis):* Captures sleep habits and resting heart rate.
   * *PC2 (Physical Activity Axis):* Captures daily steps and activity duration.
   * *Note: These 2 components retained approximately 84% of the original data's variance.*
3. **Predictive Modeling:** Trained a Multiple Linear Regression model on the extracted principal components using an 80/20 train-test split.

## 🏆 Results
The model achieved strong predictive performance and high interpretability:
* **RMSE:** ~0.79 (Predictions deviate by less than 1 unit on the 1-10 stress scale)
* **R-squared Score:** ~0.80 (The model explains 80% of the variance in stress levels)

## 💡 Business Value
This analytical pipeline demonstrates the ability to handle highly correlated (multicollinear) datasets, extract latent behavioral features, and build stable, interpretable predictive models—skills highly relevant for **customer behavior modeling, risk analysis, and operational data science.**
