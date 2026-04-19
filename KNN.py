"""
================================================================================
        K-NEAREST NEIGHBORS (KNN) - Complete Implementation
================================================================================

This module covers:
1. KNN from scratch (manual implementation)
2. KNN with sklearn
3. Distance metrics (Euclidean, Manhattan, Minkowski, Cosine)
4. Distance-weighted KNN
5. KNN for Classification & Regression
6. Finding optimal K with Cross-Validation
7. Curse of Dimensionality demonstration
8. Efficiency comparison

Author: ML Revision Series
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors
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
    make_moons,
    load_breast_cancer
)
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# PART 1: KNEIGHBORS FROM SCRATCH (MANUAL IMPLEMENTATION)
# =============================================================================

class KNeighborsClassifierFromScratch:
    """
    Manual implementation of K-Nearest Neighbors Classifier.
    
    A lazy learning algorithm that classifies based on majority vote
    of K nearest neighbors.
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Fit the model (stores training data - lazy learning!).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _compute_distance(self, x1, x2):
        """
        Compute distance between two points.
        
        Parameters:
        -----------
        x1, x2 : array-like
            Points to compute distance between
            
        Returns:
        --------
        float: Distance
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'minkowski':
            p = 2
            return np.sum(np.abs(x1 - x2) ** p) ** (1/p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _get_weights(self, distances):
        """
        Compute weights for neighbors based on distance.
        
        Parameters:
        -----------
        distances : array
            Distances to neighbors
            
        Returns:
        --------
        array: Weights
        """
        if self.weights == 'uniform':
            return np.ones(len(distances))
        elif self.weights == 'distance':
            return 1 / (distances + 1e-10)
        else:
            raise ValueError(f"Unknown weights: {self.weights}")
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        predictions = []
        
        for x_q in X:
            # Compute distances to all training points
            distances = np.array([
                self._compute_distance(x_q, x_i) 
                for x_i in self.X_train
            ])
            
            # Get indices of K nearest neighbors
            k_nearest_idx = np.argsort(distances)[:self.n_neighbors]
            
            # Get labels of K nearest neighbors
            k_labels = self.y_train[k_nearest_idx]
            
            # Get distances to K nearest neighbors
            k_distances = distances[k_nearest_idx]
            
            # Compute weights
            weights = self._get_weights(k_distances)
            
            # Weight voting
            classes = np.unique(k_labels)
            class_weights = {}
            
            for c in classes:
                mask = k_labels == c
                class_weights[c] = np.sum(weights[mask])
            
            # Return class with highest weight
            predictions.append(max(class_weights, key=class_weights.get))
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Return mean accuracy."""
        return accuracy_score(y, self.predict(X))


class KNeighborsRegressorFromScratch:
    """
    Manual implementation of K-Nearest Neighbors Regressor.
    
    Predicts value as weighted mean of K nearest neighbors.
    """
    
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Fit the model (stores training data)."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def predict(self, X):
        """
        Predict values for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        predictions = []
        
        for x_q in X:
            # Compute distances
            distances = np.array([
                np.sqrt(np.sum((x_q - x_i) ** 2)) 
                for x_i in self.X_train
            ])
            
            # Get K nearest
            k_nearest_idx = np.argsort(distances)[:self.n_neighbors]
            k_values = self.y_train[k_nearest_idx]
            k_distances = distances[k_nearest_idx]
            
            # Compute weights
            if self.weights == 'uniform':
                predictions.append(np.mean(k_values))
            else:
                w = 1 / (k_distances + 1e-10)
                weighted_mean = np.sum(w * k_values) / np.sum(w)
                predictions.append(weighted_mean)
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Return R² score."""
        return r2_score(y, self.predict(X))


# =============================================================================
# PART 2: SKLEARN IMPLEMENTATIONS
# =============================================================================

def example_knn_classification():
    """
    Example 1: KNN Classification - Iris Dataset
    """
    print("=" * 70)
    print("EXAMPLE 1: KNN Classification - Iris Dataset")
    print("=" * 70)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    species = iris.target_names
    
    print(f"\n📊 Dataset: Iris")
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   Classes: {list(species)}")
    
    # Scale features (IMPORTANT for KNN!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Test different K values
    print("\n📈 Accuracy vs K:")
    print("-" * 40)
    
    results = []
    for k in [1, 3, 5, 7, 9, 11]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((k, acc))
        print(f"   K={k:>2}: Accuracy = {acc:.4f}")
    
    # Best K
    best_k = max(results, key=lambda x: x[1])[0]
    print(f"\n✅ Best K: {best_k}")
    
    # Final model
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train, y_train)
    
    print(f"\n📋 Classification Report (K={best_k}):")
    print(classification_report(y_test, knn_best.predict(X_test), target_names=species))
    
    return knn_best, best_k


def example_knn_regression():
    """
    Example 2: KNN Regression
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: KNN Regression")
    print("=" * 70)
    
    # Generate data
    X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # KNN Regression
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    print(f"\n📊 Regression Results:")
    print(f"   MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"   R²: {r2_score(y_test, y_pred):.4f}")
    
    # Compare weights
    print("\n📊 Uniform vs Distance weighting:")
    print("-" * 40)
    
    for weights in ['uniform', 'distance']:
        knn = KNeighborsRegressor(n_neighbors=5, weights=weights)
        knn.fit(X_train, y_train)
        r2 = r2_score(y_test, knn.predict(X_test))
        print(f"   {weights:<10}: R² = {r2:.4f}")
    
    return knn


def example_find_optimal_k():
    """
    Example 3: Finding Optimal K with Cross-Validation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Finding Optimal K with Cross-Validation")
    print("=" * 70)
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    
    # Method 1: Manual search
    k_values = range(1, 31, 2)
    cv_scores = []
    
    print("\n📈 Cross-Validation Results:")
    print("-" * 50)
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    best_k_manual = k_values[np.argmax(cv_scores)]
    best_score_manual = max(cv_scores)
    
    print(f"   Best K (manual): {best_k_manual}")
    print(f"   Best Score: {best_score_manual:.4f}")
    
    # Method 2: GridSearchCV
    param_grid = {'n_neighbors': list(range(1, 31, 2))}
    
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid.fit(X, y)
    
    print(f"\n   Best K (GridSearchCV): {grid.best_params_['n_neighbors']}")
    print(f"   Best Score: {grid.best_score_:.4f}")
    
    # Visualization
    print("\n📊 CV Accuracy vs K:")
    print("-" * 50)
    for k, score in zip(k_values, cv_scores):
        bar = "█" * int(score * 40)
        print(f"   K={k:>2}: {bar} {score:.4f}")
    
    return best_k_manual


def example_distance_weighted_knn():
    """
    Example 4: Distance Weighted KNN
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Distance Weighted KNN")
    print("=" * 70)
    
    # Generate non-linear data
    X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Test weighting schemes
    print("\n📊 Weighting Schemes:")
    print("-" * 50)
    
    weights_list = [
        ('uniform', 'uniform'),
        ('distance (1/d)', 'distance'),
        ('custom Gaussian', lambda d: np.exp(-d**2 / 2)),
    ]
    
    for name, weights in weights_list:
        knn = KNeighborsClassifier(n_neighbors=5, weights=weights)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        print(f"   {name:<20}: Accuracy = {acc:.4f}")
    
    # Show how distance weighting helps
    print("\n📈 Distance weighting helps when:")
    print("   • Nearer neighbors should have more influence")
    print("   • Data has varying density across classes")
    print("   • Classes are not well-separated")


def example_curse_of_dimensionality():
    """
    Example 5: Curse of Dimensionality Demonstration
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Curse of Dimensionality")
    print("=" * 70)
    
    # Generate data with increasing dimensions
    dimensions = [2, 5, 10, 20, 50, 100, 200]
    accuracies = []
    
    print("\n📈 Accuracy vs Dimensions:")
    print("-" * 50)
    
    for dim in dimensions:
        X, y = make_classification(
            n_samples=500,
            n_features=dim,
            n_informative=min(dim, 10),
            random_state=42
        )
        
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        accuracies.append(scores.mean())
        
        print(f"   d={dim:>3}: Accuracy = {scores.mean():.4f}")
    
    print("\n📊 Observations:")
    print("   • Accuracy typically decreases as dimensions increase")
    print("   • Distances become similar across all points")
    print("   • 'Nearest neighbor' becomes less meaningful")
    
    print("\n💡 Solutions:")
    print("   • Dimensionality reduction (PCA, feature selection)")
    print("   • Feature scaling")
    print("   • Use appropriate distance metric")
    
    return dimensions, accuracies


def example_distance_metrics():
    """
    Example 6: Different Distance Metrics
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Different Distance Metrics")
    print("=" * 70)
    
    # Load iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
    
    # Test metrics
    print("\n📊 Distance Metrics:")
    print("-" * 50)
    
    metrics = [
        ('euclidean', 2),
        ('manhattan', 1),
        ('chebyshev', np.inf),
        ('minkowski p=1.5', 1.5),
        ('minkowski p=3', 3),
    ]
    
    for name, p in metrics:
        knn = KNeighborsClassifier(n_neighbors=5, p=p, metric='minkowski')
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        print(f"   {name:<20}: Accuracy = {acc:.4f}")
    
    print("\n📈 When to use each:")
    print("   • Euclidean: Default, good for general use")
    print("   • Manhattan: Less sensitive to outliers")
    print("   • Chebyshev: When one dimension matters most")
    print("   • Minkowski: Flexible, generalizes L1/L2")


def example_efficient_algorithms():
    """
    Example 7: KNN for Large Datasets
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Efficient KNN for Large Datasets")
    print("=" * 70)
    
    # Generate large dataset
    X, y = make_classification(n_samples=10000, n_features=50, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Test algorithms
    print("\n📊 Algorithm Efficiency:")
    print("-" * 50)
    
    algorithms = ['ball_tree', 'kd_tree', 'brute', 'auto']
    
    for algo in algorithms:
        knn = KNeighborsClassifier(n_neighbors=5, algorithm=algo, n_jobs=1)
        
        # Fit time
        start = time.time()
        knn.fit(X_train, y_train)
        fit_time = time.time() - start
        
        # Predict time (on test set)
        start = time.time()
        y_pred = knn.predict(X_test)
        predict_time = time.time() - start
        
        print(f"   {algo:<10}: fit={fit_time:.4f}s, predict={predict_time:.4f}s")
    
    print("\n📈 Recommendations:")
    print("   • Low dimensions (<20): Use 'kd_tree' or 'ball_tree'")
    print("   • High dimensions: Use 'brute' or 'auto'")
    print("   • Very large datasets: Use approximate methods (LSH)")


def example_manual_vs_sklearn():
    """
    Example 8: Compare Manual vs Sklearn Implementation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Manual vs Sklearn KNN")
    print("=" * 70)
    
    # Load iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
    
    # Manual implementation
    print("\n🔧 Manual KNN...")
    knn_manual = KNeighborsClassifierFromScratch(n_neighbors=5)
    knn_manual.fit(X_train, y_train)
    acc_manual = knn_manual.score(X_test, y_test)
    print(f"   Accuracy: {acc_manual:.4f}")
    
    # Sklearn implementation
    print("🔧 Sklearn KNN...")
    knn_sklearn = KNeighborsClassifier(n_neighbors=5)
    knn_sklearn.fit(X_train, y_train)
    acc_sklearn = accuracy_score(y_test, knn_sklearn.predict(X_test))
    print(f"   Accuracy: {acc_sklearn:.4f}")
    
    # Compare
    print(f"\n📊 Difference: {abs(acc_manual - acc_sklearn):.6f}")
    print("   (Small differences due to tie-breaking)")
    
    return knn_manual, knn_sklearn


# =============================================================================
# PART 3: ADVANCED EXAMPLES
# =============================================================================

def example_knn_for_breast_cancer():
    """
    Example 9: KNN for Breast Cancer Prediction
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 9: KNN for Breast Cancer (Practical Application)")
    print("=" * 70)
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"\n📊 Dataset: Breast Cancer")
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   Classes: {list(np.unique(y))} (0=malignant, 1=benign)")
    
    # Scale (critical for KNN!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal K
    print("\n📈 Finding optimal K...")
    
    best_k = 5
    best_score = 0
    
    for k in range(1, 21, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_k = k
    
    print(f"   Optimal K: {best_k}")
    print(f"   CV Accuracy: {best_score:.4f}")
    
    # Final model
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y
    )
    
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train, y_train)
    
    y_pred = knn_final.predict(X_test)
    
    print(f"\n📋 Test Set Performance:")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
    print(f"   False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
    
    return knn_final


# =============================================================================
# PART 4: VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_decision_boundary():
    """
    Visualize KNN decision boundaries.
    """
    print("\n" + "=" * 70)
    print("VISUALIZATION: KNN Decision Boundary")
    print("=" * 70)
    
    # Generate 2D data
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    # Predict
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    print(f"\n📊 Decision Boundary Created")
    print(f"   Grid: {xx.shape}")
    print(f"   Training Accuracy: {knn.score(X, y):.4f}")


def visualize_k_effect():
    """
    Visualize effect of K on decision boundary.
    """
    print("\n" + "=" * 70)
    print("VISUALIZATION: Effect of K on Decision Boundary")
    print("=" * 70)
    
    # Generate data
    X, y = make_classification(n_samples=200, n_features=2, random_state=42)
    
    # Train with different K
    print("\n📊 K Effect on Decision Boundary:")
    print("-" * 50)
    
    for k in [1, 3, 5, 11]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        acc = knn.score(X, y)
        
        # Compute complexity (would be visualized)
        print(f"   K={k:>2}: Training Accuracy = {acc:.4f}")
    
    print("\n📈 Observations:")
    print("   • K=1: Most flexible, may overfit")
    print("   • K=3-5: Balanced")
    print("   • K>11: May underfit")


# =============================================================================
# PART 5: SUMMARY
# =============================================================================

def print_summary():
    """Print summary of KNN."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                   K-NEAREST NEIGHBORS (KNN) SUMMARY                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   ALGORITHM:                                                          ║
║   Find K nearest neighbors and predict based on their labels.                 ║
║                                                                       ║
║   KEY PARAMETERS:                                                     ║
║   • n_neighbors (K): Number of neighbors to consider                    ║
║   • weights: 'uniform' or 'distance'                                 ║
║   • metric: Distance measure (euclidean, manhattan, minkowski)         ║
║                                                                       ║
║   PROS:                                                             ║
║   ✓ Simple to understand                                              ║
║   ✓ No explicit training (lazy learning)                                 ║
║   ✓ Works with any decision boundary                                     ║
║   ✓ Natural for multi-class                                        ║
║                                                                       ║
║   CONS:                                                             ║
║   ✗ Slow prediction (computes all distances)                         ║
║   ✗ Sensitive to irrelevant features                                ║
║   ✗ Curse of dimensionality                                       ║
║   ✗ Requires feature scaling                                        ║
║                                                                       ║
║   BEST FOR:                                                          ║
║   • Low-dimensional data                                           ║
║   • Complex decision boundaries                                  ║
║   • When interpretability is important                           ║
║                                                                       ║
║   TIPS:                                                              ║
║   • Scale features before using KNN!                                ║
║   • Use cross-validation to find optimal K                        ║
║   • Use odd K to avoid ties                                      ║
║   • Use distance weighting for better results                   ║
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
║           K-NEAREST NEIGHBORS (KNN) - Complete Implementation           ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all examples
    print("\n" + "▶" * 35)
    print(" RUNNING ALL EXAMPLES")
    print("▶" * 35)
    
    # Example 1: Classification
    example_knn_classification()
    
    # Example 2: Regression
    example_knn_regression()
    
    # Example 3: Finding optimal K
    example_find_optimal_k()
    
    # Example 4: Distance weighting
    example_distance_weighted_knn()
    
    # Example 5: Curse of dimensionality
    example_curse_of_dimensionality()
    
    # Example 6: Distance metrics
    example_distance_metrics()
    
    # Example 7: Efficient algorithms
    example_efficient_algorithms()
    
    # Example 8: Manual vs sklearn
    example_manual_vs_sklearn()
    
    # Example 9: Practical application
    example_knn_for_breast_cancer()
    
    # Visualizations
    visualize_decision_boundary()
    visualize_k_effect()
    
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

def quick_knn_classify(X_train, y_train, X_test, n_neighbors=5):
    """
    Quick KNN classification.
    
    Usage:
    ------
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> y_pred = quick_knn_classify(X_train, y_train, X_test, k=5)
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)


def quick_knn_regress(X_train, y_train, X_test, n_neighbors=5):
    """
    Quick KNN regression.
    
    Usage:
    ------
    >>> y_pred = quick_knn_regress(X_train, y_train, X_test, k=5)
    """
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X_train, y_train)
    return knn.predict(X_test)


def find_best_k(X, y, k_range=range(1, 31, 2)):
    """
    Find best K using cross-validation.
    
    Usage:
    ------
    >>> best_k = find_best_k(X, y)
    >>> print(f"Best K: {best_k}")
    """
    best_k = 5
    best_score = 0
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_k = k
    
    return best_k


# =============================================================================
# END OF FILE
# =============================================================================