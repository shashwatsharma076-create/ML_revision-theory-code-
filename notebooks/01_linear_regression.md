# Linear Regression

## Table of Contents
1. [Introduction](#introduction)
2. [Types of Linear Regression](#types)
3. [The Hypothesis Function](#hypothesis)
4. [Cost Function](#cost-function)
5. [Gradient Descent](#gradient-descent)
6. [Assumptions](#assumptions)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Regularization](#regularization)
9. [Feature Scaling](#feature-scaling)
10. [Model Interpretation](#interpretation)
11. [Practice Exercises](#exercises)

---

## 1. Introduction <a name="introduction"></a>

Linear regression is a **supervised learning** algorithm used for predicting continuous target variables based on one or more input features. It finds the best-fitting line (or hyperplane) that minimizes the difference between predicted and actual values.

**Use Cases:**
- House price prediction
- Sales forecasting
- Risk assessment
- Trend analysis

---

## 2. Types of Linear Regression <a name="types"></a>

### Simple Linear Regression
- One input feature (x)
- Equation: `y = mx + b`

```python
# Simple Linear Regression
# y = slope * x + intercept
```

### Multiple Linear Regression
- Multiple input features (x₁, x₂, ..., xₙ)
- Equation: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

```python
# Multiple Linear Regression
# y = intercept + coef1*x1 + coef2*x2 + ...
```

---

## 3. The Hypothesis Function <a name="hypothesis"></a>

The hypothesis function represents our prediction model:

**Simple:**
```
h(x) = θ₀ + θ₁x
```

**Multiple:**
```
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

Where:
- θ₀ = intercept (bias term)
- θ₁, θ₂, ..., θₙ = coefficients (weights)
- x = features

---

## 4. Cost Function <a name="cost-function"></a>

The cost function measures how well our model fits the data. We use **Mean Squared Error (MSE)**:

```
J(θ) = (1/2m) * Σ(h(xᵢ) - yᵢ)²
```

Where:
- m = number of training examples
- h(xᵢ) = predicted value
- yᵢ = actual value

**Why MSE?**
- Penalizes larger errors more than smaller ones
- Always positive
- Differentiable

---

## 5. Gradient Descent <a name="gradient-descent"></a>

An optimization algorithm to minimize the cost function:

```
θⱼ := θⱼ - α * ∂J(θ)/∂θⱼ
```

Where:
- α = learning rate
- ∂J(θ)/∂θⱼ = partial derivative of cost function

**Key Points:**
- Learning rate too small → slow convergence
- Learning rate too large → may diverge
- Feature scaling helps gradient descent converge faster

---

## 6. Assumptions <a name="assumptions"></a>

Linear regression relies on these assumptions:

| Assumption | Description | How to Check |
|------------|-------------|--------------|
| **Linearity** | Relationship between X and Y is linear | Scatter plot |
| **Independence** | Residuals are independent | Durbin-Watson test |
| **Homoscedasticity** | Constant variance of residuals | Residuals vs fitted plot |
| **Normality** | Residuals are normally distributed | Q-Q plot |
| **No Multicollinearity** | Features are not highly correlated | VIF, correlation matrix |

---

## 7. Evaluation Metrics <a name="evaluation-metrics"></a>

### Mean Squared Error (MSE)
```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```

### Root Mean Squared Error (RMSE)
```
RMSE = √MSE
```

### Mean Absolute Error (MAE)
```
MAE = (1/n) * Σ|yᵢ - ŷᵢ|
```

### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)
```
Where:
- SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
- SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)

**R² Interpretation:**
- 1.0 = perfect prediction
- 0.0 = constant model (predicts mean)
- Can be negative if model is worse than mean

### Adjusted R²
```
R²_adj = 1 - (1-R²)(n-1)/(n-p-1)
```
Where:
- n = number of samples
- p = number of features

---

## 8. Regularization <a name="regularization"></a>

Regularization prevents overfitting by adding a penalty term.

### Ridge Regression (L2)
```
J(θ) = MSE + λ * Σθⱼ²
```
- Shrinks coefficients but doesn't set them to zero
- Good when all features might be relevant

### Lasso Regression (L1)
```
J(θ) = MSE + λ * Σ|θⱼ|
```
- Can set some coefficients to zero (feature selection)
- Good for sparse models

### Elastic Net
Combination of Ridge and Lasso:
```
J(θ) = MSE + λ₁ * Σ|θⱼ| + λ₂ * Σθⱼ²
```

---

## 9. Feature Scaling <a name="feature-scaling"></a>

Essential for gradient descent and regularized models.

### Standardization
```
x' = (x - μ) / σ
```
- Mean = 0, Std = 1

### Min-Max Normalization
```
x' = (x - x_min) / (x_max - x_min)
```
- Range: [0, 1]

---

## 10. Model Interpretation <a name="interpretation"></a>

### Coefficient Interpretation
- **Positive coefficient**: Increasing this feature increases the target
- **Negative coefficient**: Increasing this feature decreases the target
- **Magnitude**: Size of effect per unit change

### Statistical Significance
- **p-value**: Probability coefficient is due to chance
- **95% CI**: Confidence interval for coefficient
- Target: p-value < 0.05

---

## 11. Practice Exercises <a name="exercises"></a>

### Exercise 1: Simple Linear Regression
Predict house prices based on square footage.

### Exercise 2: Multiple Linear Regression
Predict salary based on experience, education, and location.

### Exercise 3: Regularized Regression
Compare Ridge, Lasso, and Elastic Net on a dataset with multicollinearity.

### Exercise 4: Feature Engineering
Create polynomial features and compare performance.

### Exercise 5: Diagnostics
Check all assumptions and interpret the results.

---

## Quick Reference

```
┌─────────────────────────────────────────────────┐
│           LINEAR REGRESSION CHECKLIST           │
├─────────────────────────────────────────────────┤
│ □ Define problem and target variable            │
│ □ Load and explore data                         │
│ □ Check for missing values                      │
│ □ Feature engineering                           │
│ □ Handle categorical variables                  │
│ □ Scale features (if using gradient descent)   │
│ □ Split into train/test sets                    │
│ □ Train model                                   │
│ □ Evaluate with multiple metrics                │
│ □ Check assumptions                             │
│ □ Tune hyperparameters (if regularizing)         │
│ □ Interpret results                             │
└─────────────────────────────────────────────────┘
```

---

## Common Pitfalls

1. **Not checking assumptions** - Always visualize residuals
2. **Forgetting feature scaling** - Essential for gradient descent
3. **Multicollinearity** - Check correlation matrix
4. **Overfitting** - Use regularization or reduce features
5. **Data leakage** - Scale features after splitting
6. **Ignoring outliers** - They heavily influence the line

---

## When to Use Linear Regression

**✅ Good for:**
- Linear relationships
- Interpretability is important
- Baseline model
- Quick prototyping

**❌ Not good for:**
- Non-linear relationships (use polynomial or other models)
- Classification problems (use logistic regression)
- Complex patterns (use tree-based or neural networks)
