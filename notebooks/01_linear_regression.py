"""
Linear Regression - Comprehensive Implementation
=================================================

This module covers all aspects of linear regression:
1. Simple Linear Regression
2. Multiple Linear Regression
3. Gradient Descent from Scratch
4. Normal Equation
5. Regularization (Ridge, Lasso, Elastic Net)
6. Polynomial Regression
7. Model Evaluation & Diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    SGDRegressor, BayesianRidge
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.datasets import make_regression, load_diabetes
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# SECTION 1: SIMPLE LINEAR REGRESSION
# =============================================================================

def simple_linear_regression():
    """
    Simple Linear Regression: y = mx + b
    One feature, one target
    """
    print("=" * 60)
    print("SECTION 1: SIMPLE LINEAR REGRESSION")
    print("=" * 60)
    
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(f"\nCoefficient (slope): {model.coef_[0][0]:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Simple Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model


# =============================================================================
# SECTION 2: MULTIPLE LINEAR REGRESSION
# =============================================================================

def multiple_linear_regression():
    """
    Multiple Linear Regression: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
    Multiple features, one target
    """
    print("\n" + "=" * 60)
    print("SECTION 2: MULTIPLE LINEAR REGRESSION")
    print("=" * 60)
    
    X, y = make_regression(
        n_samples=500, n_features=5, noise=10, random_state=42
    )
    
    feature_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.4f}")
    print(f"\nIntercept: {model.intercept_:.4f}")
    
    print(f"\nMSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    plt.figure(figsize=(10, 5))
    plt.barh(coef_df['Feature'], coef_df['Coefficient'])
    plt.xlabel('Coefficient Value')
    plt.title('Feature Coefficients - Multiple Linear Regression')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    return model


# =============================================================================
# SECTION 3: GRADIENT DESCENT FROM SCRATCH
# =============================================================================

def gradient_descent_scratch():
    """
    Implement Linear Regression using Gradient Descent from scratch
    """
    print("\n" + "=" * 60)
    print("SECTION 3: GRADIENT DESCENT FROM SCRATCH")
    print("=" * 60)
    
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]
    
    learning_rate = 0.1
    n_iterations = 1000
    theta = np.random.randn(2, 1)
    
    print(f"\nLearning Rate: {learning_rate}")
    print(f"Iterations: {n_iterations}")
    
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - learning_rate * gradients
        
        if iteration % 200 == 0:
            cost = (1/m) * np.sum((X_b.dot(theta) - y) ** 2)
            print(f"Iteration {iteration}: Cost = {cost:.4f}")
    
    print(f"\nFinal Coefficients:")
    print(f"  Intercept (θ₀): {theta[0][0]:.4f}")
    print(f"  Slope (θ₁): {theta[1][0]:.4f}")
    
    X_test = np.array([[0], [2]])
    X_test_b = np.c_[np.ones((2, 1)), X_test]
    y_pred = X_test_b.dot(theta)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.6, label='Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Fitted Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Gradient Descent Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return theta


# =============================================================================
# SECTION 4: NORMAL EQUATION
# =============================================================================

def normal_equation():
    """
    Solve Linear Regression using the Normal Equation
    θ = (XᵀX)⁻¹Xᵀy
    No need for feature scaling, no iterative process
    """
    print("\n" + "=" * 60)
    print("SECTION 4: NORMAL EQUATION")
    print("=" * 60)
    
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]
    
    theta_normal = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    print(f"\nCoefficients (Normal Equation):")
    print(f"  Intercept (θ₀): {theta_normal[0][0]:.4f}")
    print(f"  Slope (θ₁): {theta_normal[1][0]:.4f}")
    
    X_test = np.array([[0], [2]])
    X_test_b = np.c_[np.ones((2, 1)), X_test]
    y_pred = X_test_b.dot(theta_normal)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.6, label='Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Fitted Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Normal Equation Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return theta_normal


# =============================================================================
# SECTION 5: RIDGE REGRESSION (L2 REGULARIZATION)
# =============================================================================

def ridge_regression():
    """
    Ridge Regression adds L2 penalty: λ * Σθⱼ²
    Prevents overfitting, shrinks coefficients
    """
    print("\n" + "=" * 60)
    print("SECTION 5: RIDGE REGRESSION (L2)")
    print("=" * 60)
    
    X, y = make_regression(
        n_samples=200, n_features=10, noise=20, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    train_scores = []
    test_scores = []
    
    print("\nAlpha tuning for Ridge Regression:")
    print("-" * 40)
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        
        train_score = r2_score(y_train, ridge.predict(X_train_scaled))
        test_score = r2_score(y_test, ridge.predict(X_test_scaled))
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        print(f"Alpha: {alpha:6.3f} | Train R²: {train_score:.4f} | Test R²: {test_score:.4f}")
    
    best_alpha = alphas[np.argmax(test_scores)]
    best_ridge = Ridge(alpha=best_alpha)
    best_ridge.fit(X_train_scaled, y_train)
    
    print(f"\nBest Alpha: {best_alpha}")
    print(f"Best Test R²: {max(test_scores):.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.semilogx(alphas, train_scores, 'b-o', label='Train R²')
    plt.semilogx(alphas, test_scores, 'r-o', label='Test R²')
    plt.axvline(x=best_alpha, color='g', linestyle='--', label=f'Best α={best_alpha}')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('R² Score')
    plt.title('Ridge Regression: Alpha vs R² Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_ridge, scaler


# =============================================================================
# SECTION 6: LASSO REGRESSION (L1 REGULARIZATION)
# =============================================================================

def lasso_regression():
    """
    Lasso Regression adds L1 penalty: λ * Σ|θⱼ|
    Can set coefficients to zero (feature selection)
    """
    print("\n" + "=" * 60)
    print("SECTION 6: LASSO REGRESSION (L1)")
    print("=" * 60)
    
    X, y = make_regression(
        n_samples=200, n_features=10, noise=20, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)
    
    y_pred = lasso.predict(X_test_scaled)
    
    print(f"\nLasso Regression (alpha=0.1):")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    coef_df = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(10)],
        'Coefficient': lasso.coef_
    })
    
    nonzero = coef_df[coef_df['Coefficient'] != 0]
    zero = coef_df[coef_df['Coefficient'] == 0]
    
    print(f"\nNon-zero coefficients: {len(nonzero)}")
    print(f"Zero coefficients (excluded): {len(zero)}")
    
    plt.figure(figsize=(12, 5))
    colors = ['red' if c == 0 else 'blue' for c in coef_df['Coefficient']]
    plt.bar(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    plt.xlabel('Feature')
    plt.ylabel('Coefficient Value')
    plt.title('Lasso Regression: Feature Selection (Zero vs Non-zero)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    return lasso


# =============================================================================
# SECTION 7: ELASTIC NET
# =============================================================================

def elastic_net_regression():
    """
    Elastic Net combines L1 and L2: λ₁ * Σ|θⱼ| + λ₂ * Σθⱼ²
    """
    print("\n" + "=" * 60)
    print("SECTION 7: ELASTIC NET")
    print("=" * 60)
    
    X, y = make_regression(
        n_samples=200, n_features=10, noise=20, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic.fit(X_train_scaled, y_train)
    
    y_pred = elastic.predict(X_test_scaled)
    
    print(f"\nElastic Net (alpha=0.1, l1_ratio=0.5):")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Non-zero coefficients: {np.sum(elastic.coef_ != 0)}")
    
    return elastic


# =============================================================================
# SECTION 8: POLYNOMIAL REGRESSION
# =============================================================================

def polynomial_regression():
    """
    Polynomial Regression for non-linear relationships
    """
    print("\n" + "=" * 60)
    print("SECTION 8: POLYNOMIAL REGRESSION")
    print("=" * 60)
    
    np.random.seed(42)
    X = 6 * np.random.rand(100, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)
    
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(f"\nPolynomial Regression (degree=2):")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    X_range = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_range_poly = poly_features.transform(X_range)
    y_range = model.predict(X_range_poly)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.6, label='Data')
    plt.plot(X_range, y_range, color='red', linewidth=2, label='Polynomial Fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression (Degree 2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model, poly_features


# =============================================================================
# SECTION 9: MODEL EVALUATION & DIAGNOSTICS
# =============================================================================

def model_diagnostics():
    """
    Comprehensive model evaluation and diagnostics
    """
    print("\n" + "=" * 60)
    print("SECTION 9: MODEL EVALUATION & DIAGNOSTICS")
    print("=" * 60)
    
    X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted (Homoscedasticity)')
    axes[0, 0].grid(True, alpha=0.3)
    
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normality)')
    
    axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual')
    axes[1, 1].scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('Actual vs Predicted')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nMetrics Summary:")
    print("-" * 40)
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    _, p_value = stats.normaltest(residuals)
    print(f"\nNormality Test p-value: {p_value:.4f}")
    print(f"Residuals normally distributed: {'Yes' if p_value > 0.05 else 'No'}")


# =============================================================================
# SECTION 10: CROSS-VALIDATION
# =============================================================================

def cross_validation_demo():
    """
    K-Fold Cross Validation
    """
    print("\n" + "=" * 60)
    print("SECTION 10: CROSS-VALIDATION")
    print("=" * 60)
    
    X, y = make_regression(n_samples=500, n_features=5, noise=20, random_state=42)
    
    model = LinearRegression()
    
    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='r2'
    )
    
    print(f"\n5-Fold Cross-Validation R² Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    
    print(f"\nMean R²: {cv_scores.mean():.4f}")
    print(f"Std R²: {cv_scores.std():.4f}")
    print(f"95% CI: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, "
          f"{cv_scores.mean() + 1.96*cv_scores.std():.4f}]")
    
    return cv_scores


# =============================================================================
# SECTION 11: DIABETES DATASET EXAMPLE
# =============================================================================

def diabetes_example():
    """
    Real-world example using sklearn diabetes dataset
    """
    print("\n" + "=" * 60)
    print("SECTION 11: DIABETES DATASET EXAMPLE")
    print("=" * 60)
    
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    feature_names = diabetes.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    print(f"\nDiabetes Dataset Results:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print(f"\nFeature Importance:")
    print(coef_df.to_string(index=False))
    
    plt.figure(figsize=(12, 6))
    plt.barh(coef_df['Feature'], coef_df['Coefficient'])
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.title('Diabetes: Feature Coefficients')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    return model


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Run all sections
    """
    print("\n" + "=" * 60)
    print("    LINEAR REGRESSION COMPREHENSIVE GUIDE")
    print("=" * 60)
    
    simple_linear_regression()
    multiple_linear_regression()
    gradient_descent_scratch()
    normal_equation()
    ridge_regression()
    lasso_regression()
    elastic_net_regression()
    polynomial_regression()
    model_diagnostics()
    cross_validation_demo()
    diabetes_example()
    
    print("\n" + "=" * 60)
    print("    ALL SECTIONS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
