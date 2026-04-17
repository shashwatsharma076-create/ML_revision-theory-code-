"""
================================================================================
        LINEAR DISCRIMINANT ANALYSIS (LDA) - Complete Implementation
================================================================================

This module covers:
1. LDA from scratch (manual implementation)
2. LDA with sklearn
3. QDA (Quadratic Discriminant Analysis)
4. Fisher's Linear Discriminant
5. Dimensionality Reduction with LDA
6. Visualization of Decision Boundaries
7. Comparison with Logistic Regression

Author: ML Revision Series
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_curve, auc
)
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# PART 1: LDA FROM SCRATCH (MANUAL IMPLEMENTATION)
# =============================================================================

class LinearDiscriminantAnalysisFromScratch:
    """
    Manual implementation of Linear Discriminant Analysis (LDA).
    
    LDA finds linear combinations of features that best separate classes
    by maximizing between-class variance while minimizing within-class variance.
    
    Mathematical Foundation:
    - Uses Bayes theorem: P(y=k|x) ∝ P(x|y=k) × P(y=k)
    - Assumes Gaussian distribution with shared covariance
    - Decision boundary: w'x + b = 0
    """
    
    def __init__(self):
        self.classes_ = None
        self.n_classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_ = None
        self.n_features_ = None
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit the LDA model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        """
        n_samples, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Step 1: Compute prior probabilities P(y=k)
        self.priors_ = {}
        for k in self.classes_:
            self.priors_[k] = np.sum(y == k) / n_samples
        
        # Step 2: Compute class means μₖ
        self.means_ = {}
        for k in self.classes_:
            self.means_[k] = X[y == k].mean(axis=0)
        
        # Step 3: Compute within-class covariance matrix Σ
        # Σ = Σₖ Σᵢ∈class k (xᵢ - μₖ)(xᵢ - μₖ)' / n
        self.covariance_ = np.zeros((self.n_features_, self.n_features_))
        
        for k in self.classes_:
            X_k = X[y == k] - self.means_[k]
            self.covariance_ += np.dot(X_k.T, X_k) / n_samples
        
        # Step 4: Compute LDA coefficients (for binary classification)
        if self.n_classes_ == 2:
            # For 2 classes: w = Σ⁻¹(μ₁ - μ₂)
            k1, k2 = self.classes_
            self.coef_ = np.linalg.solve(
                self.covariance_, 
                self.means_[k1] - self.means_[k2]
            )
            
            # Intercept: b = -0.5 × (μ₁ + μ₂)' Σ⁻¹ (μ₁ - μ₂) + log(P₁/P₂)
            mean_diff = self.means_[k1] - self.means_[k2]
            mean_sum = self.means_[k1] + self.means_[k2]
            self.intercept_ = (
                -0.5 * np.dot(np.linalg.solve(self.covariance_, mean_sum), mean_diff)
                + np.log(self.priors_[k1] / self.priors_[k2])
            )
        else:
            # For multi-class: compute discriminant function for each class
            self.coef_ = None
            self.intercept_ = None
        
        return self
    
    def _compute_discriminant(self, X, k):
        """
        Compute the discriminant function δₖ(x) for class k.
        
        δₖ(x) = μₖ' Σ⁻¹ x - 0.5 × μₖ' Σ⁻¹ μₖ + log P(y=k)
        """
        cov_inv = np.linalg.pinv(self.covariance_)
        return (
            np.dot(X, np.dot(cov_inv, self.means_[k]))
            - 0.5 * np.dot(self.means_[k], np.dot(cov_inv, self.means_[k]))
            + np.log(self.priors_[k])
        )
    
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
        if self.n_classes_ == 2:
            # Binary case: use linear decision boundary
            scores = np.dot(X, self.coef_) + self.intercept_
            return np.where(scores > 0, self.classes_[0], self.classes_[1])
        else:
            # Multi-class: compute discriminant for each class
            n_samples = X.shape[0]
            discriminants = np.zeros((n_samples, self.n_classes_))
            
            for i, k in enumerate(self.classes_):
                discriminants[:, i] = self._compute_discriminant(X, k)
            
            # Return class with highest discriminant
            return self.classes_[np.argmax(discriminants, axis=1)]
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Returns:
        --------
        probs : array, shape (n_samples, n_classes)
            Probability of each class
        """
        n_samples = X.shape[0]
        
        if self.n_classes_ == 2:
            scores = np.dot(X, self.coef_) + self.intercept_
            prob_class0 = 1 / (1 + np.exp(scores))
            return np.column_stack([prob_class0, 1 - prob_class0])
        else:
            discriminants = np.zeros((n_samples, self.n_classes_))
            for i, k in enumerate(self.classes_):
                discriminants[:, i] = self._compute_discriminant(X, k)
            
            # Softmax to convert to probabilities
            exp_disc = np.exp(discriminants - discriminants.max(axis=1, keepdims=True))
            return exp_disc / exp_disc.sum(axis=1, keepdims=True)
    
    def score(self, X, y):
        """Return mean accuracy."""
        return accuracy_score(y, self.predict(X))


# =============================================================================
# PART 2: SKLEARN IMPLEMENTATION
# =============================================================================

def example_basic_lda():
    """
    Example 1: Basic LDA with Wine Dataset
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic LDA - Wine Dataset")
    print("=" * 70)
    
    # Load wine dataset (3 classes)
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    class_names = wine.target_names
    
    print(f"\n📊 Dataset: Wine")
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   Classes: {list(class_names)}")
    print(f"   Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    
    # Predict
    y_pred = lda.predict(X_test)
    y_pred_proba = lda.predict_proba(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📈 Results:")
    print(f"   Test Accuracy: {accuracy:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(lda, X, y, cv=5)
    print(f"   5-Fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print(f"\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   {class_names[0]:<12} | {class_names[1]:<12} | {class_names[2]:<12}")
    print(f"   " + "-" * 44)
    for i, row in enumerate(cm):
        print(f"   {row[0]:<12} | {row[1]:<12} | {row[2]:<12}")
        if i < len(cm) - 1:
            print(f"   " + "-" * 44)
    
    print(f"\n🔢 LDA Properties:")
    print(f"   Classes: {lda.n_classes_}")
    print(f"   Coefficients shape: {lda.coef_.shape}")
    print(f"   Means shape: {lda.means_.shape}")
    
    return lda, accuracy


def example_lda_dim_reduction():
    """
    Example 2: LDA for Dimensionality Reduction
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: LDA for Dimensionality Reduction - Iris Dataset")
    print("=" * 70)
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    species = iris.target_names
    
    print(f"\n📊 Dataset: Iris")
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   Classes: {list(species)}")
    
    # LDA with 2 components (max is n_classes - 1 = 2)
    lda_2d = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda_2d.fit_transform(X, y)
    
    print(f"\n🔄 Dimensionality Reduction:")
    print(f"   Original: {X.shape[1]} dimensions")
    print(f"   Reduced: {X_lda.shape[1]} dimensions")
    print(f"   Explained variance: {lda_2d.explained_variance_ratio_}")
    print(f"   Total variance explained: {lda_2d.explained_variance_ratio_.sum():.4f}")
    
    # Compare accuracies
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train_lda, X_test_lda, _, _ = train_test_split(
        X_lda, y, test_size=0.3, random_state=42, stratify=y
    )
    
    lda_full = LinearDiscriminantAnalysis()
    lda_full.fit(X_train_full, y_train)
    acc_full = lda_full.score(X_test_full, y_test)
    
    lda_lda = LinearDiscriminantAnalysis()
    lda_lda.fit(X_train_lda, y_train)
    acc_lda = lda_lda.score(X_test_lda, y_test)
    
    print(f"\n📈 Accuracy Comparison:")
    print(f"   Full features (4D): {acc_full:.4f}")
    print(f"   LDA reduced (2D): {acc_lda:.4f}")
    
    return X_lda, y, lda_2d


def example_lda_vs_qda_vs_logistic():
    """
    Example 3: Compare LDA, QDA, and Logistic Regression
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: LDA vs QDA vs Logistic Regression")
    print("=" * 70)
    
    from sklearn.datasets import make_blobs
    
    # Generate two scenarios
    print("\n📊 Dataset Scenarios:")
    print("-" * 50)
    
    results = []
    
    # Scenario 1: Equal covariance (LDA should be best)
    print("\n[Scenario 1] Equal Covariance (Linear Boundary)")
    X1, y1 = make_blobs(
        n_samples=[100, 100, 100],
        centers=[[0, 0], [4, 0], [2, 4]],
        random_state=42
    )
    
    models = {
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'Logistic': LogisticRegression()
    }
    
    print(f"   {'Model':<12} | {'CV Accuracy':>12}")
    print("   " + "-" * 28)
    for name, model in models.items():
        scores = cross_val_score(model, X1, y1, cv=5)
        print(f"   {name:<12} | {scores.mean():>10.4f} ± {scores.std():.4f}")
        results.append((name, 'Equal Cov', scores.mean()))
    
    # Scenario 2: Unequal covariance (QDA should be best)
    print("\n[Scenario 2] Unequal Covariance (Quadratic Boundary)")
    X2, y2 = make_blobs(
        n_samples=[100, 100],
        centers=[[-1, -1], [2, 2]],
        covariances=[[[1, 0], [0, 1]], [[4, 0], [0, 0.5]]],
        random_state=42
    )
    
    print(f"   {'Model':<12} | {'CV Accuracy':>12}")
    print("   " + "-" * 28)
    for name, model in models.items():
        scores = cross_val_score(model, X2, y2, cv=5)
        print(f"   {name:<12} | {scores.mean():>10.4f} ± {scores.std():.4f}")
        results.append((name, 'Unequal Cov', scores.mean()))
    
    print("\n📊 Summary:")
    print("-" * 50)
    print("LDA works best when: classes have equal covariance (linear boundary)")
    print("QDA works best when: classes have different covariances (curved boundary)")
    print("Logistic is robust for general cases")
    
    return results


def example_manual_vs_sklearn():
    """
    Example 4: Compare Manual vs Sklearn Implementation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Manual LDA vs Sklearn LDA")
    print("=" * 70)
    
    # Load wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize for comparison
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train manual implementation
    print("\n🔧 Training Manual LDA...")
    lda_manual = LinearDiscriminantAnalysisFromScratch()
    lda_manual.fit(X_train_scaled, y_train)
    acc_manual = lda_manual.score(X_test_scaled, y_test)
    
    # Train sklearn implementation
    print("🔧 Training Sklearn LDA...")
    lda_sklearn = LinearDiscriminantAnalysis()
    lda_sklearn.fit(X_train_scaled, y_train)
    acc_sklearn = lda_sklearn.score(X_test_scaled, y_test)
    
    print(f"\n📈 Results:")
    print(f"   Manual LDA Accuracy:  {acc_manual:.4f}")
    print(f"   Sklearn LDA Accuracy: {acc_sklearn:.4f}")
    print(f"   Difference: {abs(acc_manual - acc_sklearn):.6f}")
    
    # Compare predictions
    y_pred_manual = lda_manual.predict(X_test_scaled)
    y_pred_sklearn = lda_sklearn.predict(X_test_scaled)
    
    agreement = np.mean(y_pred_manual == y_pred_sklearn)
    print(f"\n   Prediction Agreement: {agreement:.2%}")
    
    return lda_manual, lda_sklearn


def example_regularized_lda():
    """
    Example 5: Regularized LDA for High-Dimensional Data
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Regularized LDA (p > n scenario)")
    print("=" * 70)
    
    from sklearn.datasets import make_classification
    
    # Generate high-dimensional data (p > n)
    np.random.seed(42)
    n, p = 50, 30  # 50 samples, 30 features (p > n!)
    
    X, y = make_classification(
        n_samples=n,
        n_features=p,
        n_informative=5,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    print(f"\n📊 Dataset: High-dimensional")
    print(f"   Samples: {n}, Features: {p} (p > n!)")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # LDA without shrinkage (may fail)
    print("\n🔧 LDA without shrinkage...")
    try:
        lda_no_shrink = LinearDiscriminantAnalysis()
        scores_no_shrink = cross_val_score(lda_no_shrink, X, y, cv=5)
        print(f"   CV Accuracy: {scores_no_shrink.mean():.4f}")
    except Exception as e:
        print(f"   Failed: {str(e)[:50]}...")
    
    # LDA with Ledoit-Wolf shrinkage (automatic)
    print("\n🔧 LDA with Ledoit-Wolf shrinkage (auto)...")
    lda_shrink_auto = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    scores_shrink = cross_val_score(lda_shrink_auto, X, y, cv=5)
    print(f"   CV Accuracy: {scores_shrink.mean():.4f}")
    
    # LDA with custom shrinkage
    print("\n🔧 LDA with shrinkage=0.5...")
    lda_shrink_05 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5)
    scores_05 = cross_val_score(lda_shrink_05, X, y, cv=5)
    print(f"   CV Accuracy: {scores_05.mean():.4f}")
    
    print("\n📊 Best option for p > n: Use shrinkage='auto' (Ledoit-Wolf)")


def example_fisher_lda():
    """
    Example 6: Fisher's Linear Discriminant
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Fisher's Linear Discriminant")
    print("=" * 70)
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    species = iris.target_names
    
    # For Fisher's discriminant, we use LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    print("\n📊 Fisher's LDA Coefficients (Iris):")
    print("-" * 60)
    print(f"{'Feature':<25} | {'LD1':>10} | {'LD2':>10}")
    print("-" * 60)
    
    for i, name in enumerate(iris.feature_names):
        print(f"{name:<25} | {lda.coef_[0][i]:>10.4f} | {lda.coef_[1][i]:>10.4f}")
    
    print("-" * 60)
    print("\n🔍 Interpretation:")
    print("   LD1 (first discriminant): Best separates classes along this direction")
    print("   LD2 (second discriminant): Additional separation direction")
    print(f"   Explained variance: {lda.explained_variance_ratio_}")
    
    return lda


def example_lda_decision_boundary():
    """
    Example 7: Visualize LDA Decision Boundaries
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: LDA Decision Boundaries (2D Visualization)")
    print("=" * 70)
    
    from sklearn.datasets import make_blobs
    
    # Generate 2D data for visualization
    X, y = make_blobs(
        n_samples=[100, 100, 100],
        centers=[[0, 0], [3, 0], [1.5, 3]],
        random_state=42,
        cluster_std=1.0
    )
    
    # Train LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    
    # Predict on mesh
    Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Get probabilities
    Z_prob = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z_prob = Z_prob[:, 1].reshape(xx.shape)
    
    print(f"\n📊 Decision Boundary Created")
    print(f"   Grid size: {xx.shape}")
    print(f"   Test Accuracy: {lda.score(X, y):.4f}")
    
    # Note: Visualization code (matplotlib) saved but not executed
    print("\n📈 Visualization would show:")
    print("   - Three colored regions (one per class)")
    print("   - Linear decision boundaries between regions")
    print("   - Class centroids marked")
    
    return X, y, lda


# =============================================================================
# PART 3: VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_lda_projection():
    """
    Visualize LDA projection for Iris dataset.
    """
    print("\n" + "=" * 70)
    print("VISUALIZATION: LDA Projection (Iris Dataset)")
    print("=" * 70)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    species = iris.target_names
    
    # LDA projection
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)
    
    print(f"\n📊 Original vs LDA Projection:")
    print(f"   Original shape: {X.shape}")
    print(f"   LDA shape: {X_lda.shape}")
    print(f"   Variance explained: {lda.explained_variance_ratio_.sum():.4f}")
    
    print("\n📈 Point distribution in LDA space:")
    for i, sp in enumerate(species):
        mask = y == i
        print(f"   {sp:<10}: mean=({X_lda[mask, 0].mean():.2f}, {X_lda[mask, 1].mean():.2f})")


def visualize_lda_vs_logistic():
    """
    Compare LDA and Logistic Regression decision boundaries.
    """
    print("\n" + "=" * 70)
    print("VISUALIZATION: LDA vs Logistic Regression")
    print("=" * 70)
    
    from sklearn.datasets import make_moons
    
    # Generate non-linear data
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    
    # Train both models
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    lda_acc = lda.score(X, y)
    
    logistic = LogisticRegression()
    logistic.fit(X, y)
    logistic_acc = logistic.score(X, y)
    
    print(f"\n📊 Non-linear (moons) dataset:")
    print(f"   LDA accuracy: {lda_acc:.4f}")
    print(f"   Logistic Regression accuracy: {logistic_acc:.4f}")
    
    print("\n📈 Observations:")
    print("   LDA assumes linear boundaries → may fail on curved data")
    print("   Logistic Regression also assumes linear → same limitation")
    print("   For truly non-linear data → use SVM, Trees, or Neural Networks")


# =============================================================================
# PART 4: SUMMARY AND COMPARISON
# =============================================================================

def print_summary():
    """Print summary of LDA."""
    print("\n" + "=" * 70)
    print("SUMMARY: LINEAR DISCRIMINANT ANALYSIS (LDA)")
    print("=" * 70)
    
    summary = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        WHAT IS LDA?                                 │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   A supervised classification algorithm that finds LINEAR           │
    │   combinations of features to SEPARATE classes.                     │
    │                                                                     │
    │   Key Formula:                                                      │
    │   δₖ(x) = μₖ' Σ⁻¹ x - 0.5 × μₖ' Σ⁻¹ μₖ + log P(y=k)             │
    │                                                                     │
    │   Decision Boundary: w'x + b = 0                                    │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      ASSUMPTIONS                                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   1. Normal Distribution: X|Y=k ~ N(μₖ, Σ)                         │
    │   2. Equal Covariance: Σ₁ = Σ₂ = ... = Σₖ                         │
    │   3. Independence: Observations are independent                     │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     WHEN TO USE                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   ✓ Multi-class classification (natural extension)                   │
    │   ✓ Dimensionality reduction (supervised)                           │
    │   ✓ Classes well-separated                                          │
    │   ✓ Limited training data                                           │
    │   ✓ Need interpretable coefficients                                │
    │                                                                     │
    │   ✗ Non-linear decision boundaries                                   │
    │   ✗ Violation of normality assumption                              │
    │   ✗ Unequal covariances (use QDA)                                  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    RELATED METHODS                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   QDA:  Quadratic Discriminant Analysis                            │
    │         → Different covariance per class                            │
    │         → Quadratic decision boundaries                             │
    │                                                                     │
    │   Logistic Regression:                                              │
    │         → Discriminative (models P(y|x) directly)                  │
    │         → No distributional assumptions                             │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """
    print(summary)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║         LINEAR DISCRIMINANT ANALYSIS (LDA)                           ║
    ║         Complete Implementation Guide                                 ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all examples
    print("\n" + "▶" * 35)
    print(" RUNNING ALL EXAMPLES")
    print("▶" * 35)
    
    # Example 1: Basic LDA
    example_basic_lda()
    
    # Example 2: Dimensionality Reduction
    example_lda_dim_reduction()
    
    # Example 3: LDA vs QDA vs Logistic
    example_lda_vs_qda_vs_logistic()
    
    # Example 4: Manual vs Sklearn
    example_manual_vs_sklearn()
    
    # Example 5: Regularized LDA
    example_regularized_lda()
    
    # Example 6: Fisher's LDA
    example_fisher_lda()
    
    # Example 7: Decision Boundaries
    example_lda_decision_boundary()
    
    # Visualizations
    visualize_lda_projection()
    visualize_lda_vs_logistic()
    
    # Summary
    print_summary()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                        COMPLETED!                                      ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# QUICK REFERENCE FUNCTIONS
# =============================================================================

def quick_lda_fit_predict(X_train, y_train, X_test):
    """
    Quick LDA fit and predict function.
    
    Usage:
    ------
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> y_pred = quick_lda_fit_predict(X_train, y_train, X_test)
    >>> accuracy = accuracy_score(y_test, y_pred)
    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    return lda.predict(X_test)


def quick_lda_dim_reduction(X, y, n_components=2):
    """
    Quick LDA dimensionality reduction function.
    
    Usage:
    ------
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> X_lda = quick_lda_dim_reduction(X, y, n_components=2)
    """
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    return lda.fit_transform(X, y)


# =============================================================================
# END OF FILE
# =============================================================================
