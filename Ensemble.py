"""
================================================================================
                    ENSEMBLE METHODS - Complete Implementation
================================================================================

This module covers:
1. Random Forest (Classification & Regression)
2. AdaBoost
3. Gradient Boosting
4. Voting Classifier
5. Stacking
6. Bagging
7. Feature Importance

Author: ML Revision Series
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor
)
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_squared_error
)
from sklearn.datasets import (
    make_classification,
    make_regression
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# PART 1: RANDOM FOREST CLASSIFICATION
# =============================================================================

def example_random_forest():
    """
    Example 1: Random Forest Classification
    """
    print("=" * 70)
    print("EXAMPLE 1: Random Forest Classification")
    print("=" * 70)
    
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print(f"\n📊 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Number of trees: {clf.n_estimators}")
    print(f"   Max features per split: {clf.max_features}")
    print(f"   Feature importance (top 5): {clf.feature_importances_[:5]}")
    
    # Vary number of trees
    print("\n📊 Number of Trees Effect:")
    print("-" * 40)
    for n in [10, 50, 100, 200]:
        clf = RandomForestClassifier(n_estimators=n, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"   n_estimators={n:<4}: {acc:.4f}")


# =============================================================================
# PART 2: RANDOM FOREST REGRESSION
# =============================================================================

def example_random_forest_regression():
    """
    Example 2: Random Forest Regression
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Random Forest Regression")
    print("="*70)
    
    X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    
    print(f"\n📊 Results:")
    print(f"   R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"   MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")


# =============================================================================
# PART 3: ADABOOST
# =============================================================================

def example_adaboost():
    """
    Example 3: AdaBoost
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: AdaBoost")
    print("="*70)
    
    X, y = make_classification(n_samples=500, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    clf = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        random_state=42,
        algorithm='SAMME'
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print(f"\n📊 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Estimators: {clf.n_estimators}")
    print(f"   Learning rate: {clf.learning_rate}")
    print(f"   Feature importance: {clf.feature_importances_[:5]}")
    
    # Vary learning rate
    print("\n📊 Learning Rate Effect:")
    print("-" * 40)
    for lr in [0.01, 0.1, 0.5, 1.0, 2.0]:
        clf = AdaBoostClassifier(n_estimators=50, learning_rate=lr, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"   learning_rate={lr:<4}: {acc:.4f}")


# =============================================================================
# PART 4: GRADIENT BOOSTING
# =============================================================================

def example_gradient_boosting():
    """
    Example 4: Gradient Boosting
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Gradient Boosting")
    print("="*70)
    
    X, y = make_classification(n_samples=500, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print(f"\n📊 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Stages: {clf.n_estimators}")
    print(f"   Learning rate: {clf.learning_rate}")
    print(f"   Max depth: {clf.max_depth}")
    
    # Show training stages
    scores = clf.score(X_test, y_test)
    print(f"\n📊 Training Progression:")
    for i in [10, 25, 50, 75, 100]:
        if i <= clf.n_estimators:
            stage_score = list(clf.staged_score(X_test, y_test))
            print(f"   Stage {i:<3}: {stage_score[min(i-1, len(stage_score)-1)]:.4f}")


# =============================================================================
# PART 5: VOTING CLASSIFIER
# =============================================================================

def example_voting():
    """
    Example 5: Voting Classifier
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Voting Classifier")
    print("="*70)
    
    X, y = make_classification(n_samples=500, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Hard voting
    clf_hard = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=200)),
            ('dt', DecisionTreeClassifier()),
            ('svc', SVC())
        ],
        voting='hard'
    )
    clf_hard.fit(X_train, y_train)
    acc_hard = accuracy_score(y_test, clf_hard.predict(X_test))
    
    # Soft voting
    clf_soft = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=200)),
            ('dt', DecisionTreeClassifier()),
            ('svc', SVC(probability=True))
        ],
        voting='soft'
    )
    clf_soft.fit(X_train, y_train)
    acc_soft = accuracy_score(y_test, clf_soft.predict(X_test))
    
    print(f"\n📊 Comparison:")
    print(f"   Hard Voting: {acc_hard:.4f}")
    print(f"   Soft Voting: {acc_soft:.4f}")
    
    # Individual models
    print(f"\n📊 Individual Models:")
    for name, model in clf_hard.named_estimators_.items():
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"   {name:<6}: {acc:.4f}")


# =============================================================================
# PART 6: STACKING
# =============================================================================

def example_stacking():
    """
    Example 6: Stacking Classifier
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Stacking Classifier")
    print("="*70)
    
    X, y = make_classification(n_samples=500, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Stacking with different base models
    clf = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=200)),
            ('dt', DecisionTreeClassifier()),
            ('rf', RandomForestClassifier(n_estimators=50))
        ],
        final_estimator=LogisticRegression(),
        cv=5,
        passthrough=False
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Stacking Accuracy: {acc:.4f}")
    
    # Compare with best base model
    print("\n📊 Base Models:")
    for name, model in clf.named_estimators_.items():
        base_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"   {name}<6>: {base_acc:.4f}")


# =============================================================================
# PART 7: COMPARE ALL METHODS
# =============================================================================

def example_compare():
    """
    Example 7: Compare All Ensemble Methods
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Compare All Ensemble Methods")
    print("="*70)
    
    X, y = make_classification(n_samples=500, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n📊 Comparison (n_estimators=50):")
    print("-" * 50)
    
    methods = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=50, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'Bagging': BaggingClassifier(n_estimators=50, random_state=42)
    }
    
    for name, clf in methods.items():
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"   {name:<20}: {acc:.4f}")


# =============================================================================
# PART 8: FEATURE IMPORTANCE
# =============================================================================

def example_feature_importance():
    """
    Example 8: Feature Importance
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Feature Importance - Random Forest")
    print("="*70)
    
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=5,
        n_redundant=5,
        n_repeated=5,
        random_state=42
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    importance = clf.feature_importances_
    indices = importance.argsort()[::-1]
    
    print(f"\n📊 Top 10 Features by Importance:")
    print("-" * 50)
    
    for i in range(10):
        idx = indices[i]
        bar = "█" * int(importance[idx] * 30)
        print(f"   Feature {idx:<3}: {bar} {importance[idx]:.4f}")


# =============================================================================
# PART 9: BAGGING
# =============================================================================

def example_bagging():
    """
    Example 9: Bagging vs Random Forest
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: Bagging vs Random Forest")
    print("="*70)
    
    X, y = make_classification(n_samples=500, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Bagging with Decision Tree
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    acc_bagging = accuracy_score(y_test, bagging.predict(X_test))
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"\n📊 Comparison:")
    print(f"   Bagging (DT base): {acc_bagging:.4f}")
    print(f"   Random Forest:     {acc_rf:.4f}")
    
    print(f"\n📋 Key Differences:")
    print(f"   Bagging: Random features at splits (RF adds this)")
    print(f"   RF: Also de-correlates trees via feature subset")


# =============================================================================
# PART 10: EXTRA TREES
# =============================================================================

def example_extra_trees():
    """
    Example 10: Extra Trees vs Random Forest
    """
    print("\n" + "="*70)
    print("EXAMPLE 10: Extra Trees vs Random Forest")
    print("="*70)
    
    X, y = make_classification(n_samples=500, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    
    # Extra Trees
    et = ExtraTreesClassifier(n_estimators=50, random_state=42)
    et.fit(X_train, y_train)
    acc_et = accuracy_score(y_test, et.predict(X_test))
    
    print(f"\n📊 Comparison:")
    print(f"   Random Forest: {acc_rf:.4f}")
    print(f"   Extra Trees:  {acc_et:.4f}")
    
    print(f"\n📋 Extra Trees Key:")
    print(f"   - Random splits (not best split)")
    print(f"   - Faster than RF")
    print(f"   - More variance, potentially better")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary():
    """Print ensemble summary."""
    print("""
╔════════════════════════���═���════════════════════════════════════════════════════╗
║                       ENSEMBLE METHODS SUMMARY                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   TYPES:                                                            ║
║   • Random Forest: Bagging + feature randomness                       ║
║   • Extra Trees: Random splits, no best split                          ║
║   • AdaBoost: Reweight samples, fix errors                            ║
║   • Gradient Boosting: Optimize loss function                       ║
║   • Voting: Combine model predictions                               ║
║   • Stacking: Meta-learner on base predictions                     ║
║   • Bagging: Bootstrap + aggregation                                 ║
║                                                                       ║
║   KEY PARAMETERS:                                                     ║
║   • n_estimators: Number of base models                             ║
║   • max_depth: Maximum tree depth (RF uses None)                 ║
║   • learning_rate: Shrinkage factor (boosting)                     ║
║   • max_features: Features per split                                  ║
║                                                                       ║
║   PROS:                                                             ║
║   ✓ Reduces variance and bias                                         ║
║   ✓ Handles complex patterns                                         ║
║   ✓ Feature importance                                             ║
║   ✓ Robust to outliers                                              ║
║                                                                       ║
║   CONS:                                                             ║
║   ✗ Slower than single models                                       ║
║   ✗ Less interpretable                                             ║
║   ✗ More memory                                                    ║
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
║               ENSEMBLE METHODS - Complete Implementation               ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n" + "▶" * 35)
    print(" RUNNING ALL EXAMPLES")
    print("▶" * 35)
    
    # Examples
    example_random_forest()
    example_random_forest_regression()
    example_adaboost()
    example_gradient_boosting()
    example_voting()
    example_stacking()
    example_compare()
    example_feature_importance()
    example_bagging()
    example_extra_trees()
    
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

def quick_random_forest(X_train, y_train, X_test, n_estimators=100):
    """
    Quick Random Forest classification.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def quick_gradient_boosting(X_train, y_train, X_test, n_estimators=100):
    """
    Quick Gradient Boosting classification.
    """
    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


# =============================================================================
# END OF FILE
# =============================================================================