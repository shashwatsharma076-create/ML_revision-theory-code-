"""
================================================================================
                    NAIVE BAYES - Complete Implementation
================================================================================

This module covers:
1. Gaussian Naive Bayes
2. Multinomial Naive Bayes
3. Bernoulli Naive Bayes
4. Complement Naive Bayes
5. Text Classification
6. Comparison

Author: ML Revision Series
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB
)
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
from sklearn.datasets import (
    load_iris,
    load_wine,
    make_classification
)
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# PART 1: GAUSSIAN NAIVE BAYES
# =============================================================================

def example_gaussian_iris():
    """
    Example 1: Gaussian Naive Bayes - Iris Dataset
    """
    print("=" * 70)
    print("EXAMPLE 1: Gaussian Naive Bayes - Iris Dataset")
    print("=" * 70)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"\nDataset: Iris ({X.shape[0]} samples, {X.shape[1]} features)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print(f"\n📊 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Show class priors
    print(f"\n📊 Class Priors:")
    for i, prior in enumerate(clf.class_prior_):
        print(f"   {iris.target_names[i]}: {prior:.4f}")
    
    # Show means for first class
    print(f"\n📊 Feature Means (class 0):")
    for i, (name, mean) in enumerate(zip(iris.feature_names, clf.theta_[0])):
        print(f"   {name:<20}: {mean:.4f}")


# =============================================================================
# PART 2: MULTINOMIAL NAIVE BAYES
# =============================================================================

def example_multinomial_text():
    """
    Example 2: Multinomial Naive Bayes - Text Classification
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multinomial Naive Bayes - Text Classification")
    print("="*70)
    
    # Sample data
    texts = [
        "Chinese Beijing Chinese",
        "Chinese Chinese Shanghai",
        "Chinese Macao",
        "Tokyo Japan Chinese",
        "Japanese",
        "Tokyo Yokohama"
    ]
    labels = [1, 1, 1, 0, 0, 0]
    
    # Vectorize
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    
    print(f"\n📊 Vocabulary: {vectorizer.get_feature_names()}")
    print(f"   Feature matrix: {X.shape}")
    
    # Train
    clf = MultinomialNB(alpha=1.0)  # Smoothing
    clf.fit(X, labels)
    
    # Predictions
    test_texts = [
        "Chinese Tokyo",
        "Japan China",
        "Chinese"
    ]
    
    print(f"\n📊 Test Predictions:")
    for text in test_texts:
        X_test = vectorizer.transform([text])
        pred = clf.predict(X_test)[0]
        proba = clf.predict_proba(X_test)[0]
        
        label_name = "China" if pred == 1 else "Japan"
        print(f"   '{text}' → {label_name}")
        print(f"      P(Japan)={proba[0]:.4f}, P(China)={proba[1]:.4f}")


# =============================================================================
# PART 3: BERNOULLI NAIVE BAYES
# =============================================================================

def example_bernoulli():
    """
    Example 3: Bernoulli Naive Bayes
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Bernoulli Naive Bayes")
    print("="*70)
    
    X = [
        [1, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    y = [1, 1, 1, 0, 0, 0]
    
    clf = BernoulliNB()
    clf.fit(X, y)
    
    print(f"\n📊 Training Complete")
    print(f"   Class priors: {clf.class_log_prior_}")
    
    # Test
    test_samples = [
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    
    print(f"\n📊 Test Predictions:")
    for sample in test_samples:
        pred = clf.predict([sample])[0]
        proba = clf.predict_proba([sample])[0]
        print(f"   {sample} → class {pred} (P={proba})")


# =============================================================================
# PART 4: COMPLEMENT NAIVE BAYES
# =============================================================================

def example_complement():
    """
    Example 4: Complement Naive Bayes - Imbalanced Data
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Complement Naive Bayes - Imbalanced Data")
    print("="*70)
    
    # Imbalanced data (90% class 0, 10% class 1)
    X, y = make_classification(
        n_samples=1000,
        weights=[0.9, 0.1],
        random_state=42
    )
    
    print(f"\n📊 Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"   Class {u}: {c} samples")
    
    # Compare classifiers
    print("\n📊 Comparison (5-fold CV):")
    print("-" * 50)
    
    classifiers = [
        ("Gaussian", GaussianNB()),
        ("Multinomial", MultinomialNB()),
        ("Complement", ComplementNB())
    ]
    
    for name, clf in classifiers:
        scores = cross_val_score(clf, X, y, cv=5)
        print(f"   {name:<12}: {scores.mean():.4f} (+/- {scores.std():.4f})")


# =============================================================================
# PART 5: COMPARISON
# =============================================================================

def example_comparison():
    """
    Example 5: Comparing All Naive Bayes Types
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Comparing All Naive Bayes Types - Wine Dataset")
    print("="*70)
    
    wine = load_wine()
    X, y = wine.data, wine.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Test different Naive Bayes types
    print("\n📊 Naive Bayes Types Comparison:")
    print("-" * 50)
    
    results = {}
    
    # Gaussian
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    results["Gaussian"] = acc
    print(f"   Gaussian:    {acc:.4f}")
    
    # Multinomial (need non-negative - bin the continuous data)
    X_pos = X_train - X_train.min()
    X_pos = X_pos + 1
    clf = MultinomialNB()
    clf.fit(X_pos, y_train)
    X_test_pos = X_test - X_train.min() + 1
    acc = accuracy_score(y_test, clf.predict(X_test_pos))
    results["Multinomial"] = acc
    print(f"   Multinomial:  {acc:.4f}")
    
    # Bernoulli (binarize)
    X_bin = (X_train > X_train.mean()).astype(int)
    clf = BernoulliNB()
    clf.fit(X_bin, y_train)
    X_test_bin = (X_test > X_train.mean()).astype(int)
    acc = accuracy_score(y_test, clf.predict(X_test_bin))
    results["Bernoulli"] = acc
    print(f"   Bernoulli:   {acc:.4f}")
    
    # Complement
    clf = ComplementNB()
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    results["Complement"] = acc
    print(f"   Complement:  {acc:.4f}")
    
    best = max(results, key=results.get)
    print(f"\n✅ Best: {best} ({results[best]:.4f})")


# =============================================================================
# PART 6: PROBABILITY OUTPUT
# =============================================================================

def example_probabilities():
    """
    Example 6: Probability Estimates
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Probability Estimates")
    print("="*70)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    # Get probabilities for first test sample
    sample = X_test[0:1]
    pred = clf.predict(sample)[0]
    proba = clf.predict_proba(sample)[0]
    
    print(f"\n📊 Sample: {sample[0]}")
    print(f"   Predicted: {iris.target_names[pred]}")
    print(f"   Probabilities:")
    for i, p in enumerate(proba):
        bar = "█" * int(p * 20)
        print(f"      {iris.target_names[i]:<12}: {p:.4f} {bar}")
    
    # Show all test predictions with probabilities
    print(f"\n📊 First 5 Test Predictions:")
    for i in range(5):
        sample = X_test[i:i+1]
        pred = clf.predict(sample)[0]
        proba = clf.predict_proba(sample)[0]
        actual = y_test[i]
        correct = "✓" if pred == actual else "✗"
        print(f"   [{i}] Pred={pred}, Actual={actual} {correct} → {proba[:3]}")


# =============================================================================
# PART 7: SMOOTHING EFFECT
# =============================================================================

def example_smoothing():
    """
    Example 7: Laplace Smoothing
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Laplace Smoothing Effect")
    print("="*70)
    
    # Text classification with small dataset
    texts = [
        "good movie", "good film", "great movie",
        "bad movie", "terrible film"
    ]
    labels = [1, 1, 1, 0, 0]  # 1=positive, 0=negative
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    
    print(f"\n📊 Vocabulary: {vectorizer.get_feature_names()}")
    
    # Train with different smoothing
    print("\n📊 Smoothing Effect:")
    print("-" * 50)
    
    test = ["good great terrible bad"]
    X_test = vectorizer.transform(test)
    
    for alpha in [0, 0.5, 1, 10]:
        clf = MultinomialNB(alpha=alpha)
        clf.fit(X, labels)
        proba = clf.predict_proba(X_test)[0]
        
        if alpha == 0:
            status = "(no smoothing)"
        elif alpha == 1:
            status = "(Laplace)"
        else:
            status = f"(α={alpha})"
        
        print(f"   α={alpha:>2}: {status}")
        print(f"       Neg={proba[0]:.4f}, Pos={proba[1]:.4f}")


# =============================================================================
# PART 8: SPAM DETECTION EXAMPLE
# =============================================================================

def example_spam_detection():
    """
    Example 8: Spam Detection
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Spam Detection - Practical Example")
    print("="*70)
    
    # Sample data
    texts = [
        "free lottery win money",      # spam
        "congratulations you won",     # spam
        "click here to win prize",    # spam
        "meeting tomorrow at office",   # ham
        "project deadline next week",  # ham
        "see you in meeting",         # ham
        "free offer limited time",       # spam
        "your project report ready",  # ham
    ]
    labels = [1, 1, 1, 0, 0, 0, 1, 0]  # 1=spam, 0=ham
    
    # Vectorize
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X, labels)
    
    # Test new emails
    test_emails = [
        "free money win now",
        "meeting scheduled",
        "you won free prize click"
    ]
    
    print(f"\n📊 Spam Detection:")
    print("-" * 60)
    
    for email in test_emails:
        X_test = vectorizer.transform([email])
        pred = clf.predict(X_test)[0]
        proba = clf.predict_proba(X_test)[0]
        
        label = "SPAM" if pred == 1 else "HAM"
        confidence = max(proba) * 100
        
        print(f"   '{email}'")
        print(f"      → {label} ({confidence:.1f}% confidence)")
        print(f"      Ham: {proba[0]:.4f}, Spam: {proba[1]:.4f}")


# =============================================================================
# PART 9: SUMMARY
# =============================================================================

def print_summary():
    """Print summary of Naive Bayes."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       NAIVE BAYES SUMMARY                              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   ALGORITHM:                                                         ║
║   Uses Bayes theorem with naive independence assumption               ║
║   P(y|X) ∝ P(X|y) × P(y)                                            ║
║                                                                       ║
║   TYPES:                                                            ║
║   • Gaussian: Continuous data (normal distribution)                 ║
║   • Multinomial: Count data (text, word frequencies)                  ║
║   • Bernoulli: Binary features (present/absent)                       ║
║   • Complement: Imbalanced datasets                                 ║
║                                                                       ║
║   KEY PARAMETERS:                                                     ║
║   • alpha: Laplace smoothing parameter (default=1)                 ║
║                                                                       ║
║   PROS:                                                             ║
║   ✓ Very fast                                                       ║
║   ✓ Works well with small data                                        ║
║   ✓ Handles high dimensionality                                   ║
║   ✓ Provides probability output                                    ║
║   ✓ Robust to irrelevant features                                  ║
║                                                                       ║
║   CONS:                                                             ║
║   ✗ Naive assumption rarely holds                                 ║
║   ✗ Can have zero probability issues                              ║
║   ✗ Less accurate than complex models in some cases                ║
║                                                                       ║
║   BEST FOR:                                                         ║
║   • Text classification (spam, sentiment)                           ║
║   • Quick baseline models                                          ║
║   • Multi-class problems                                          ║
║   • When probability output is needed                               ║
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
║               NAIVE BAYES - Complete Implementation                     ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n" + "▶" * 35)
    print(" RUNNING ALL EXAMPLES")
    print("▶" * 35)
    
    # Example 1: Gaussian - Iris
    example_gaussian_iris()
    
    # Example 2: Multinomial - Text
    example_multinomial_text()
    
    # Example 3: Bernoulli
    example_bernoulli()
    
    # Example 4: Complement
    example_complement()
    
    # Example 5: Comparison
    example_comparison()
    
    # Example 6: Probabilities
    example_probabilities()
    
    # Example 7: Smoothing
    example_smoothing()
    
    # Example 8: Spam Detection
    example_spam_detection()
    
    # Summary
    print_summary()
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        COMPLETED!                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# QUICK REFERENCE FUNCTIONS
# =============================================================================

def quick_naive_bayes(X_train, y_train, X_test):
    """
    Quick Gaussian Naive Bayes classification.
    
    Usage:
    ------
    >>> y_pred = quick_naive_bayes(X_train, y_train, X_test)
    """
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def quick_text_classifier(texts, labels, test_texts):
    """
    Quick text classification using Multinomial NB.
    
    Usage:
    ------
    >>> preds = quick_text_classifier(train_texts, labels, test_texts)
    """
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(texts)
    X_test = vectorizer.transform(test_texts)
    
    clf = MultinomialNB()
    clf.fit(X_train, labels)
    
    return clf.predict(X_test), clf.predict_proba(X_test)


# =============================================================================
# END OF FILE
# =============================================================================