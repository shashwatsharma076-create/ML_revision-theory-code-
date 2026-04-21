"""
================================================================================
                SUPPORT VECTOR MACHINES (SVM) - Complete Implementation
================================================================================

This module covers:
1. SVM from scratch
2. SVM with sklearn
3. Kernel implementations (Linear, RBF, Polynomial)
4. Classification & Regression
5. Parameter tuning
6. Feature importance

Author: ML Revision Series
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    classification_report
)
from sklearn.datasets import (
    make_classification,
    make_regression,
    make_moons,
    load_iris
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# PART 1: SVM WITH SKLEARN
# =============================================================================

def example_linear_svm():
    """
    Example 1: Linear SVM Classification
    """
    print("=" * 70)
    print("EXAMPLE 1: Linear SVM - Classification")
    print("=" * 70)
    
    # Generate linearly separable data
    X, y = make_classification(
        n_samples=500, 
        n_features=2, 
        class_sep=2,
        random_state=42
    )
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3
    )
    
    # Scale features (IMPORTANT for SVM!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train linear SVM
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    
    print(f"\n📊 Linear SVM Results:")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Support vectors: {clf.n_support_}")
    print(f"   Total SV: {sum(clf.n_support_)}")
    
    # Compare different C values
    print("\n📊 C Parameter Effect:")
    print("-" * 40)
    
    for c in [0.01, 0.1, 1, 10]:
        clf = SVC(kernel='linear', C=c)
        clf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        n_sv = clf.n_support_.sum()
        print(f"   C={c:>5}: Accuracy = {acc:.4f}, SV = {n_sv}")


def example_rbf_kernel():
    """
    Example 2: RBF Kernel (Non-linear)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: RBF Kernel - Moon Dataset")
    print("=" * 70)
    
    # Generate moon-shaped data (non-linear)
    X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compare kernels
    print("\n📊 Kernel Comparison:")
    print("-" * 50)
    
    results = {}
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        clf = SVC(kernel=kernel)
        clf.fit(X_train_scaled, y_train)
        acc = clf.score(X_test_scaled, y_test)
        n_sv = clf.n_support_.sum()
        results[kernel] = acc
        print(f"   {kernel:<10}: Accuracy = {acc:.4f}, SV = {n_sv}")
    
    # Best kernel
    best_kernel = max(results, key=results.get)
    print(f"\n✅ Best Kernel: {best_kernel}")


def example_kernel_comparison():
    """
    Example 3: All Kernels on Same Data
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Kernel Comparison")
    print("=" * 70)
    
    # Generate data
    X, y = make_moons(n_samples=500, noise=0.15)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Test different kernels
    print("\n📊 All Kernels:")
    print("-" * 60)
    
    # Linear
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"   Linear:     Accuracy = {acc:.4f}, SV = {clf.n_support_.sum()}")
    
    # RBF with different gamma
    for gamma in [0.01, 0.1, 1, 10]:
        clf = SVC(kernel='rbf', gamma=gamma)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"   RBF(γ={gamma:>4}): Accuracy = {acc:.4f}, SV = {clf.n_support_.sum()}")
    
    # Polynomial with different degree
    for degree in [2, 3, 4]:
        clf = SVC(kernel='poly', degree=degree)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"   Poly(d={degree}):   Accuracy = {acc:.4f}, SV = {clf.n_support_.sum()}")
    
    # Sigmoid
    clf = SVC(kernel='sigmoid')
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"   Sigmoid:   Accuracy = {acc:.4f}, SV = {clf.n_support_.sum()}")


def example_rbf_gamma():
    """
    Example 4: RBF gamma parameter
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: RBF Gamma Effect")
    print("=" * 70)
    
    X, y = make_moons(n_samples=500, noise=0.1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\n📊 Gamma Effect:")
    print("-" * 50)
    
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        clf = SVC(kernel='rbf', gamma=gamma)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        n_sv = clf.n_support_.sum()
        
        # Visualize (text-based)
        if gamma <= 1:
            status = "underfit"
        elif gamma <= 10:
            status = "optimal"
        else:
            status = "overfit"
        
        print(f"   gamma={gamma:>6}: Acc = {acc:.4f}, SV = {n_sv:>3} ({status})")
    
    print("\n📈 Observations:")
    print("   - Small gamma: underfitting (too smooth)")
    print("   - Large gamma: overfitting (too complex)")
    print("   - Optimal gamma: best generalization")


def example_polynomial_kernel():
    """
    Example 5: Polynomial Kernel
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Polynomial Kernel")
    print("=" * 70)
    
    X, y = make_classification(n_samples=500, n_features=2, class_sep=1.5, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\n📊 Polynomial Degree:")
    print("-" * 40)
    
    for degree in [1, 2, 3, 4, 5]:
        clf = SVC(kernel='poly', degree=degree)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"   degree={degree}: Accuracy = {acc:.4f}")


def example_svr():
    """
    Example 6: SVM Regression
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: SVM Regression")
    print("=" * 70)
    
    # Generate regression data
    X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1)).flatten()
    
    # Train SVM regression
    reg = SVR(kernel='rbf')
    reg.fit(X_train_scaled, y_train_scaled)
    
    # Predict
    y_pred_scaled = reg.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1)).flatten()
    
    print(f"\n📊 SVM Regression Results:")
    print(f"   MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"   R²: {r2_score(y_test, y_pred):.4f}")
    
    # Compare kernels
    print("\n📊 Kernel Comparison:")
    print("-" * 40)
    
    for kernel in ['linear', 'rbf', 'poly']:
        reg = SVR(kernel=kernel)
        reg.fit(X_train_scaled, y_train_scaled)
        y_pred_s = reg.predict(X_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1)).flatten()
        r2 = r2_score(y_test, y_pred)
        print(f"   {kernel:<10}: R² = {r2:.4f}")


def example_parameter_tuning():
    """
    Example 7: Grid Search for Optimal Parameters
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Parameter Tuning with GridSearchCV")
    print("=" * 70)
    
    # Generate data
    X, y = make_classification(n_samples=500, random_state=42)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Grid search
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
    }
    
    grid = GridSearchCV(
        SVC(kernel='rbf'),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid.fit(X_train, y_train)
    
    print(f"\n✅ Best Parameters: {grid.best_params_}")
    print(f"   Best CV Score: {grid.best_score_:.4f}")
    
    # Test performance
    test_acc = grid.score(X_test, y_test)
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # Show all results
    print("\n📊 Grid Search Results:")
    print("-" * 60)
    print(f"{'C':<8} {'gamma':<10} {'CV Score':<12}")
    print("-" * 60)
    
    results = grid.cv_results_
    for i in range(len(results['params'])):
        params = results['params'][i]
        mean_score = results['mean_test_score'][i]
        print(f"{params['C']:<8} {params['gamma']:<10} {mean_score:.4f}")


def example_feature_weights():
    """
    Example 8: Linear SVM Feature Weights
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Feature Importance via Linear SVM Weights")
    print("=" * 70)
    
    # Generate data with known important features
    X, y = make_classification(
        n_samples=500, 
        n_features=10,
        n_informative=5,
        random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train linear SVM
    clf = SVC(kernel='linear')
    clf.fit(X_scaled, y)
    
    # Get weights
    weights = np.abs(clf.coef_[0])
    indices = np.argsort(weights)[::-1]
    
    print("\n📊 Top Features (by absolute weight):")
    print("-" * 50)
    
    for i in range(10):
        idx = indices[i]
        w = weights[idx]
        bar = "█" * int(w * 20)
        print(f"   Feature {idx:>2}: {bar} {w:.4f}")


def example_soft_margin():
    """
    Example 9: Hard vs Soft Margin
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Hard Margin vs Soft Margin")
    print("=" * 70)
    
    # Generate overlapping data
    X, y = make_classification(
        n_samples=500, 
        class_sep=0.5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\n📊 C Parameter Effect (Soft Margin):")
    print("-" * 50)
    
    for c in [0.001, 0.01, 0.1, 1, 10, 100]:
        clf = SVC(kernel='rbf', C=c)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        n_sv = clf.n_support_.sum()
        
        if c < 0.1:
            margin = "large (underfit)"
        elif c < 10:
            margin = "optimal"
        else:
            margin = "small (overfit)"
        
        print(f"   C={c:>7}: Acc = {acc:.4f}, SV = {n_sv:>3} ({margin})")
    
    print("\n📈 Key Insight:")
    print("   Large C → Small margin (harder, may overfit)")
    print("   Small C → Large margin (softer, may underfit)")


def example_iris_classification():
    """
    Example 10: Iris Dataset Classification
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 10: Iris Dataset - Multi-class SVM")
    print("=" * 70)
    
    # Load iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    print(f"\n📊 Multi-class SVM Results:")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Support vectors per class: {clf.n_support_}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Feature names and decision function shape
    print(f"   Decision function shape: {clf.decision_function(X_test[:1]).shape}")


# =============================================================================
# PART 2: SUMMARY
# =============================================================================

def print_summary():
    """Print SVM summary."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   SUPPORT VECTOR MACHINES SUMMARY                    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   ALGORITHM:                                                         ║
║   Finds maximum margin hyperplane to separate classes                    ║
║                                                                       ║
║   KERNELS:                                                          ║
║   • Linear: w·x + b                                                 ║
║   • Polynomial: (x·z + c)^d                                      ║
║   • RBF: exp(-γ||x-z||²)                                           ║
║   • Sigmoid: tanh(αx·z + c)                                        ║
║                                                                       ║
║   KEY PARAMETERS:                                                     ║
║   • C (regularization): Penalty for misclassification                  ║
║   • gamma: RBF kernel coefficient                                ║
║   • degree: Polynomial degree                                    ║
║                                                                       ║
║   PROS:                                                             ║
║   ✓ Effective in high dimensions                                    ║
║   ✓ Memory efficient (depends on support vectors)                ║
║   ✓ Versatile via kernels                                         ║
║   ✓ Robust to outliers                                          ║
║                                                                       ║
║   CONS:                                                             ║
║   ✗ Doesn't scale to large datasets                             ║
║   ✗ Requires feature scaling                                        ║
║   ✗ No probability estimates (needs Platt scaling)            ║
║                                                                       ║
║   TIPS:                                                              ║
║   • Always scale features!                                        ║
║   • Start with RBF kernel                                          ║
║   • Use GridSearchCV for parameters                                 ║
║   • 'scale' gamma works well for most cases                         ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║               SUPPORT VECTOR MACHINES - Complete Implementation     ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run examples
    print("\n" + "▶" * 35)
    print(" RUNNING ALL EXAMPLES")
    print("▶" * 35)
    
    # Example 1: Linear SVM
    example_linear_svm()
    
    # Example 2: RBF Kernel
    example_rbf_kernel()
    
    # Example 3: Kernel Comparison
    example_kernel_comparison()
    
    # Example 4: RBF gamma
    example_rbf_gamma()
    
    # Example 5: Polynomial
    example_polynomial_kernel()
    
    # Example 6: Regression
    example_svr()
    
    # Example 7: Parameter tuning
    example_parameter_tuning()
    
    # Example 8: Feature weights
    example_feature_weights()
    
    # Example 9: Soft margin
    example_soft_margin()
    
    # Example 10: Iris
    example_iris_classification()
    
    # Summary
    print_summary()
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        COMPLETED!                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# QUICK REFERENCE
# =============================================================================

def quick_svm_classify(X_train, y_train, X_test, 
                    kernel='rbf', C=1.0, gamma='scale'):
    """
    Quick SVM classification.
    
    Usage:
    ------
    >>> y_pred = quick_svm_classify(X_train, y_train, X_test, kernel='rbf')
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def quick_svm_regress(X_train, y_train, X_test, 
                   kernel='rbf', C=1.0, epsilon=0.1):
    """
    Quick SVM regression.
    
    Usage:
    ------
    >>> y_pred = quick_svm_regress(X_train, y_train, X_test)
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    reg = SVR(kernel=kernel, C=C, epsilon=epsilon)
    reg.fit(X_train, y_train)
    return reg.predict(X_test)


# =============================================================================
# END OF FILE
# =============================================================================