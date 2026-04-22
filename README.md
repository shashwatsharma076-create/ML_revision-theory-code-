# ML Concepts Revision

Repository for revising machine learning concepts with comprehensive code examples and theory.

## Topics

### Supervised Learning
- [Linear Regression](./notebooks/01_linear_regression.md) - Simple, Multiple, Polynomial
- [Regularization](./Regularization_Complete_Guide.txt) ✅ - Ridge, Lasso, Elastic Net
- [Logistic Regression](./Logistic_Regression_Complete_Guide.txt) ✅ - Binary, Multinomial, Ordinal
- [Linear Discriminant Analysis (LDA)](./LDA_Complete_Guide.txt) ✅ - LDA, QDA, Fisher's Discriminant
- [K-Nearest Neighbors (KNN)](./KNN_Complete_Guide.txt) ✅ - Classification, Regression, Distance Metrics
- [Decision Trees](./Decision_Trees_Complete_Guide.txt) ✅ - Entropy, Gini, Pruning, Regression
- [Support Vector Machines (SVM)](./SVM_Complete_Guide.txt) ✅ - Kernel Trick, RBF, Classification, Regression
- [Naive Bayes](./Naive_Bayes_Complete_Guide.txt) ✅ - Gaussian, Multinomial, Bernoulli, Complement
- [Random Forests](./notebooks/04_random_forests.md) (Coming Soon)

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

## Decision Trees Sub-topics ✅

1. Introduction to Decision Trees
2. How Decision Trees Work (Step-by-Step Algorithm)
3. Splitting Criteria - Entropy & Information Gain
4. Gini Index & Classification Error
5. ID3, C4.5, CART Algorithms
6. Building Decision Trees
7. Pruning Strategies (Pre-pruning, Post-pruning)
8. Handling Overfitting
9. Decision Trees for Regression
10. sklearn Implementation (Iris, Breast Cancer datasets)

## Support Vector Machines (SVM) Sub-topics ✅

1. Introduction to SVM & Maximum Margin Classifier
2. Geometric Understanding (Hyperplane, Margin, Support Vectors)
3. Mathematical Derivation (Primal & Dual Form)
4. Kernel Trick & Common Kernels (Linear, Polynomial, RBF, Sigmoid)
5. Non-Linear SVM with Kernel Mapping
6. Soft Margin SVM (Slack Variables, C parameter)
7. SVM for Regression
8. sklearn Implementation (Classification & Regression)
9. Parameter Tuning (C, gamma, degree)
10. Feature Importance via Linear SVM Weights

## Naive Bayes Sub-topics ✅

1. Introduction to Naive Bayes & Bayes Theorem
2. Naive Independence Assumption
3. Gaussian Naive Bayes (Continuous Features)
4. Multinomial Naive Bayes (Count Data/Text)
5. Bernoulli Naive Bayes (Binary Features)
6. Complement Naive Bayes (Imbalanced Data)
7. Mathematical Derivation
8. Laplace Smoothing
9. sklearn Implementation (Iris, Wine datasets)
10. Practical Text Classification (Spam Detection)