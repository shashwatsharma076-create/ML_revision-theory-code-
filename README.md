# ML Concepts Revision

Repository for revising machine learning concepts with comprehensive code examples and theory.

## Topics

### Supervised Learning
- [Linear Regression](./notebooks/01_linear_regression.md) - Simple, Multiple, Polynomial
- [Regularization](./Regularization_Complete_Guide.txt) ✅ - Ridge, Lasso, Elastic Net
- [Logistic Regression](./Logistic_Regression_Complete_Guide.txt) ✅ - Binary, Multinomial, Ordinal
- [Linear Discriminant Analysis (LDA)](./LDA_Complete_Guide.txt) ✅ - LDA, QDA, Fisher's Discriminant
- [K-Nearest Neighbors (KNN)](./KNN_Complete_Guide.txt) ✅ - Classification, Regression, Distance Metrics
- [Decision Trees](./notebooks/03_decision_trees.md) (Coming Soon)
- [Random Forests](./notebooks/04_random_forests.md) (Coming Soon)
- [Support Vector Machines](./notebooks/05_svm.md) (Coming Soon)

### Unsupervised Learning
- [Clustering](./notebooks/06_clustering.md) (Coming Soon)
- [Principal Component Analysis](./notebooks/07_pca.md) (Coming Soon)

## Quick Start

```bash
pip install -r requirements.txt
```

## Running Code

**Python Scripts:**
```bash
python notebooks/01_linear_regression.py
```

**Jupyter Notebooks:**
```bash
jupyter notebook notebooks/01_linear_regression.ipynb
```

## Linear Regression Sub-topics

1. Simple Linear Regression
2. Multiple Linear Regression
3. Gradient Descent Algorithm
4. Normal Equation
5. Regularization (Ridge, Lasso, Elastic Net) ✅
6. Polynomial Regression
7. Model Evaluation & Diagnostics
8. Cross-Validation
9. Feature Scaling
10. Real-World Examples

## Logistic Regression Sub-topics ✅

1. Binary (Binomial) Logistic Regression
2. Multinomial Logistic Regression
3. Ordinal Logistic Regression
4. Sigmoid Function & Derivation
5. Softmax Function & Derivation
6. Cost Function (Cross-Entropy)
7. Gradient Descent Derivation
8. Evaluation Metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
9. Regularization in Logistic Regression
10. sklearn Implementation

## Regularization Sub-topics ✅

1. Ridge Regression (L2) - Closed-form solution
2. Lasso Regression (L1) - Feature Selection
3. Elastic Net (L1 + L2)
4. Geometric Interpretation (L1 vs L2 constraints)
5. Bias-Variance Trade-off
6. Cross-Validation for λ selection
7. Implementation with sklearn

## Linear Discriminant Analysis (LDA) Sub-topics ✅

1. Introduction to LDA & Generative vs Discriminative Models
2. Types of Discriminant Analysis (LDA, QDA, RDA)
3. LDA Assumptions (Normality, Homoscedasticity)
4. Bayes Theorem & Posterior Probability
5. Mathematical Derivation (Step-by-Step)
6. Decision Boundaries (Linear Hyperplane)
7. Fisher's Linear Discriminant
8. LDA for Dimensionality Reduction
9. Regularized LDA with Shrinkage
10. sklearn Implementation (Wine, Iris datasets)

## K-Nearest Neighbors (KNN) Sub-topics ✅

1. Introduction to KNN (Lazy Learning)
2. How KNN Works (Step-by-Step Algorithm)
3. Distance Metrics (Euclidean, Manhattan, Minkowski, Cosine)
4. Distance-Weighted KNN
5. Derivation from Bayes Theorem
6. Choosing Optimal K (Cross-Validation)
7. Curse of Dimensionality
8. KNN for Classification (Majority Vote)
9. KNN Regression (Mean/Weighted Mean)
10. sklearn Implementation (Iris, Breast Cancer datasets)