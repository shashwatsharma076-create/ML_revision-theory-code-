"""
================================================================================
                    DECISION TREES - Complete Implementation
================================================================================

This module covers:
1. Decision Trees from scratch
2. Decision Trees with sklearn
3. Entropy & Gini implementation
4. Classification & Regression
5. Pruning strategies
6. Feature importance
7. Rule extraction

Author: ML Revision Series
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,
    export_text
)
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix
)
from sklearn.datasets import (
    load_iris,
    make_classification,
    make_regression,
    load_breast_cancer
)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# PART 1: DECISION TREE FROM SCRATCH (MANUAL IMPLEMENTATION)
# =============================================================================

class DecisionTreeClassifierFromScratch:
    """
    Manual implementation of Decision Tree Classifier.
    
    Uses Gini impurity for splitting.
    """
    
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.n_classes = None
    
    def _gini(self, y):
        """Calculate Gini impurity."""
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _best_split(self, X, y):
        """Find best split for a node."""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        current_gini = self._gini(y)
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                
                if np.sum(left_mask) < self.min_samples_split:
                    continue
                if np.sum(right_mask) < self.min_samples_split:
                    continue
                
                y_left, y_right = y[left_mask], y[right_mask]
                
                n_left, n_right = len(y_left), len(y_right)
                n = n_left + n_right
                
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                
                weighted_gini = (n_left/n * gini_left + n_right/n * gini_right)
                gain = current_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            return {'leaf': True, 'class': np.bincount(y).argmax()}
        
        # Find best split
        feature, threshold, gain = self._best_split(X, y)
        
        if feature is None or gain == 0:
            return {'leaf': True, 'class': np.bincount(y).argmax()}
        
        # Split
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def fit(self, X, y):
        """Fit the decision tree."""
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(np.array(X), np.array(y))
        return self
    
    def _predict_sample(self, x, node):
        """Predict single sample."""
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        return self._predict_sample(x, node['right'])
    
    def predict(self, X):
        """Predict class labels."""
        X = np.array(X)
        return np.array([self._predict_sample(x, self.tree) for x in X])
    
    def score(self, X, y):
        """Return mean accuracy."""
        return accuracy_score(y, self.predict(X))


# =============================================================================
# PART 2: SKLEARN IMPLEMENTATIONS
# =============================================================================

def example_classification():
    """
    Example 1: Decision Tree Classification - Iris Dataset
    """
    print("=" * 70)
    print("EXAMPLE 1: Decision Tree - Iris Dataset")
    print("=" * 70)
    
    # Load iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"\nDataset: Iris ({X.shape[0]} samples, {X.shape[1]} features)")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train with different depths
    print("\n📈 Accuracy vs Tree Depth:")
    print("-" * 40)
    
    for depth in [1, 2, 3, 5, 10, None]:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"   depth={str(depth):>4}: Accuracy = {acc:.4f}")
    
    # Feature importance
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    print("\n📊 Feature Importance:")
    for name, imp in zip(iris.feature_names, clf.feature_importances_):
        bar = "█" * int(imp * 30)
        print(f"   {name:<20}: {bar} {imp:.4f}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, clf.predict(X_test)))
    
    return clf


def example_entropy_vs_gini():
    """
    Example 2: Compare Entropy vs Gini
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Entropy vs Gini")
    print("=" * 70)
    
    # Generate data
    X, y = make_classification(n_samples=1000, random_state=42)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Compare criteria
    print("\n📊 Entropy vs Gini Comparison:")
    print("-" * 50)
    
    results = {}
    for criterion in ['gini', 'entropy']:
        clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
        
        # Training accuracy
        train_acc = accuracy_score(y_train, clf.fit(X_train, y_train).predict(X_train))
        
        # Test accuracy
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        
        # CV score
        cv_scores = cross_val_score(clf, X, y, cv=5)
        
        results[criterion] = {
            'train': train_acc,
            'test': test_acc,
            'cv': cv_scores.mean()
        }
        
        print(f"\n{criterion.upper()}:")
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print("\n📊 Note: Differences are usually negligible")


def example_pruning():
    """
    Example 3: Pruning Strategies
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Pruning Strategies")
    print("=" * 70)
    
    # Generate data
    X, y = make_classification(n_samples=500, random_state=42)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    print("\n📊 Pruning Parameters:")
    print("-" * 50)
    
    # Unpruned
    clf_unpruned = DecisionTreeClassifier(random_state=42)
    clf_unpruned.fit(X_train, y_train)
    acc_unpruned = accuracy_score(y_test, clf_unpruned.predict(X_test))
    print(f"   Unpruned: accuracy = {acc_unpruned:.4f}, leaves = {clf_unpruned.get_n_leaves()}")
    
    # Max depth
    for depth in [3, 5, 10]:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"   max_depth={depth}: accuracy = {acc:.4f}, leaves = {clf.get_n_leaves()}")
    
    # Min samples leaf
    for min leaf in [5, 10, 20]:
        clf = DecisionTreeClassifier(min_samples_leaf=min leaf, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"   min_samples_leaf={min leaf}: accuracy = {acc:.4f}, leaves = {clf.get_n_leaves()}")
    
    print("\n💡 Tip: Use GridSearchCV to find optimal parameters")


def example_regression():
    """
    Example 4: Decision Tree Regression
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Decision Tree Regression")
    print("=" * 70)
    
    # Generate data
    X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Train
    reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg.fit(X_train, y_train)
    
    # Predict
    y_pred = reg.predict(X_test)
    
    print(f"\n📊 Regression Results:")
    print(f"   MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"   R²: {r2_score(y_test, y_pred):.4f}")
    
    # Compare depths
    print("\n📈 R² vs Depth:")
    print("-" * 40)
    
    for depth in [2, 3, 5, 10, None]:
        reg = DecisionTreeRegressor(max_depth=depth)
        reg.fit(X_train, y_train)
        r2 = r2_score(y_test, reg.predict(X_test))
        print(f"   depth={str(depth):>4}: R² = {r2:.4f}")


def example_feature_importance():
    """
    Example 5: Feature Importance
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Feature Importance - Breast Cancer Dataset")
    print("=" * 70)
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Train
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    
    # Get importance
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    print(f"\n📊 Top 15 Important Features:")
    print("-" * 60)
    
    for i in range(15):
        idx = indices[i]
        print(f"   {i+1:>2}. {data.feature_names[idx]:<30}: {importance[idx]:.4f}")
    
    # Visualize
    print("\n📈 Importance Distribution:")
    print("-" * 40)
    n_important = np.sum(importance > 0.01)
    print(f"   Features with importance > 0.01: {n_important}")


def example_extract_rules():
    """
    Example 6: Extract Decision Rules
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Extract Decision Rules")
    print("=" * 70)
    
    # Train on iris
    iris = load_iris()
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(iris.data, iris.target)
    
    # Extract rules as text
    print("\n📋 Decision Rules (Text):")
    print("-" * 50)
    rules = export_text(clf, feature_names=list(iris.feature_names))
    print(rules)
    
    # Tree info
    print(f"\n📊 Tree Statistics:")
    print(f"   Max Depth: {clf.get_depth()}")
    print(f"   Number of Leaves: {clf.get_n_leaves()}")


def example_manual_vs_sklearn():
    """
    Example 7: Manual vs Sklearn Comparison
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Manual vs Sklearn Implementation")
    print("=" * 70)
    
    # Load iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Manual implementation
    print("\n🔧 Manual Decision Tree...")
    clf_manual = DecisionTreeClassifierFromScratch(max_depth=5)
    clf_manual.fit(X_train, y_train)
    acc_manual = clf_manual.score(X_test, y_test)
    print(f"   Accuracy: {acc_manual:.4f}")
    
    # Sklearn implementation
    print("🔧 Sklearn Decision Tree...")
    clf_sklearn = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf_sklearn.fit(X_train, y_train)
    acc_sklearn = accuracy_score(y_test, clf_sklearn.predict(X_test))
    print(f"   Accuracy: {acc_sklearn:.4f}")
    
    print(f"\n📊 Difference: {abs(acc_manual - acc_sklearn):.4f}")
    print("   (Manual implementation is simplified)")


def example_cross_validation():
    """
    Example 8: Hyperparameter Tuning with Cross-Validation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Hyperparameter Tuning")
    print("=" * 70)
    
    # Generate data
    X, y = make_classification(n_samples=500, random_state=42)
    
    # Grid search
    param_grid = {
        'max_depth': [3, 5, 7, None],
        'min_samples_leaf': [5, 10, 20],
        'min_samples_split': [10, 20, 30],
    }
    
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid.fit(X, y)
    
    print(f"\n✅ Best Parameters: {grid.best_params_}")
    print(f"   Best CV Score: {grid.best_score_:.4f}")
    
    # Top 5 combinations
    print("\n📊 Top 5 Parameter Combinations:")
    print("-" * 60)
    
    results = pd.DataFrame(grid.cv_results_)
    results = results.sort_values('rank_test_score')
    
    for i in range(min(5, len(results))):
        row = results.iloc[i]
        params = row['params']
        score = row['mean_test_score']
        print(f"   {params}: {score:.4f}")


# =============================================================================
# PART 3: ADVANCED EXAMPLES
# =============================================================================

def visualize_tree_structure():
    """
    Visualize Decision Tree Structure
    """
    print("\n" + "=" * 70)
    print("VISUALIZATION: Decision Tree Structure")
    print("=" * 70)
    
    # Train on iris
    iris = load_iris()
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(iris.data, iris.target)
    
    print(f"\n📊 Tree Properties:")
    print(f"   Max Depth: {clf.get_depth()}")
    print(f"   Number of Leaves: {clf.get_n_leaves()}")
    print(f"   Number of Features: {clf.n_features_in_}")
    
    # Feature importance
    importance = clf.feature_importances_
    total_importance = np.sum(importance > 0)
    
    print(f"\n   Important Features: {total_importance}")
    print(f"   Sum of Importance: {np.sum(importance):.4f}")


# =============================================================================
# PART 4: SUMMARY
# =============================================================================

def print_summary():
    """Print summary of Decision Trees."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       DECISION TREES SUMMARY                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   ALGORITHM:                                                         ║
║   Recursively split data based on features to maximize homogeneity            ║
║                                                                       ║
║   KEY PARAMETERS:                                                     ║
║   • max_depth: Maximum depth of tree                                     ║
║   • min_samples_leaf: Min samples in leaf node                       ║
║   • min_samples_split: Min samples to split                          ║
║   • criterion: 'gini' or 'entropy'                               ║
║                                                                       ║
║   PROS:                                                             ║
║   ✓ Easy to interpret and visualize                                 ║
║   ✓ Handles both categorical and numerical features                  ║
║   ✓ No feature scaling needed                                    ║
║   ✓ Can capture non-linear relationships                         ║
║                                                                       ║
║   CONS:                                                             ║
║   ✗ Can overfit deep trees                                      ║
║   ✗ Unstable with small data changes                           ║
║   ✗ Can be biased with imbalanced classes                     ║
║                                                                       ║
║   BEST FOR:                                                         ║
║   • Interpretable models needed                                  ║
║   • Feature importance analysis                                 ║
║   • As base for ensemble methods (Random Forest)                  ║
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
║                 DECISION TREES - Complete Implementation                    ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all examples
    print("\n" + "▶" * 35)
    print(" RUNNING ALL EXAMPLES")
    print("▶" * 35)
    
    # Example 1: Classification
    example_classification()
    
    # Example 2: Entropy vs Gini
    example_entropy_vs_gini()
    
    # Example 3: Pruning
    example_pruning()
    
    # Example 4: Regression
    example_regression()
    
    # Example 5: Feature Importance
    example_feature_importance()
    
    # Example 6: Extract Rules
    example_extract_rules()
    
    # Example 7: Manual vs Sklearn
    example_manual_vs_sklearn()
    
    # Example 8: Cross-Validation
    example_cross_validation()
    
    # Visualization
    visualize_tree_structure()
    
    # Summary
    print_summary()
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        COMPLETED!                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# QUICK REFERENCE FUNCTIONS
# =============================================================================

def quick_decision_tree(X_train, y_train, X_test, max_depth=5):
    """
    Quick Decision Tree classification.
    
    Usage:
    ------
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> y_pred = quick_decision_tree(X_train, y_train, X_test, max_depth=5)
    """
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def quick_decision_tree_regressor(X_train, y_train, X_test, max_depth=5):
    """
    Quick Decision Tree regression.
    
    Usage:
    ------
    >>> y_pred = quick_decision_tree_regressor(X_train, y_train, X_test, max_depth=5)
    """
    reg = DecisionTreeRegressor(max_depth=max_depth)
    reg.fit(X_train, y_train)
    return reg.predict(X_test)


# =============================================================================
# END OF FILE
# =============================================================================