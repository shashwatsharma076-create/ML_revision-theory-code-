import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("        PRINCIPAL COMPONENT ANALYSIS (PCA) - 10 EXAMPLES")
print("="*70)

# ============================================================================
# EXAMPLE 1: Basic PCA on Iris Dataset
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 1: Basic PCA - Iris Dataset")
print("="*70)

iris = load_iris()
X, y = iris.data, iris.target

print(f"Original shape: {X.shape} (samples x features)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nExplained Variance Ratio per Component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"   PC{i+1}: {var:.4f} ({var*100:.2f}%)")

print(f"\nCumulative Explained Variance:")
cumvar = np.cumsum(pca.explained_variance_ratio_)
for i, cv in enumerate(cumvar):
    print(f"   {i+1} component(s): {cv:.4f} ({cv*100:.2f}%)")

# ============================================================================
# EXAMPLE 2: Dimensionality Reduction
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 2: Dimensionality Reduction")
print("="*70)

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

print(f"Original dimensions: 4")
print(f"Reduced to: {X_2d.shape[1]} dimensions")
print(f"Variance retained: {sum(pca_2d.explained_variance_ratio_)*100:.2f}%")

# ============================================================================
# EXAMPLE 3: Optimal Components (95% Variance)
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 3: Finding Optimal Components")
print("="*70)

pca_95 = PCA(n_components=0.95)
X_95 = pca_95.fit_transform(X_scaled)

print(f"Components needed for 95% variance: {pca_95.n_components_}")
print(f"Actual variance explained: {sum(pca_95.explained_variance_ratio_)*100:.2f}%")

# ============================================================================
# EXAMPLE 4: PCA for Visualization
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 4: PCA for Visualization")
print("="*70)

X_blob, y_blob = make_blobs(n_samples=300, centers=3, n_features=5, 
                           random_state=42, cluster_std=2)

scaler_blob = StandardScaler()
X_blob_scaled = scaler_blob.fit_transform(X_blob)

pca_vis = PCA(n_components=2)
X_blob_2d = pca_vis.fit_transform(X_blob_scaled)

print(f"5D data → 2D for visualization")
print(f"Variance explained: {sum(pca_vis.explained_variance_ratio_)*100:.2f}%")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap='viridis', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original 2D Projection')

plt.subplot(1, 2, 2)
plt.scatter(X_blob_2d[:, 0], X_blob_2d[:, 1], c=y_blob, cmap='viridis', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('After PCA (2D)')
plt.tight_layout()
plt.savefig('pca_visualization.png', dpi=100)
plt.close()
print("Saved: pca_visualization.png")

# ============================================================================
# EXAMPLE 5: PCA as Preprocessing in Pipeline
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 5: PCA in ML Pipeline")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression(max_iter=200))
])

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)

print(f"Pipeline: StandardScaler → PCA(2) → LogisticRegression")
print(f"Test accuracy: {score:.4f}")

# ============================================================================
# EXAMPLE 6: Compare With/Without PCA
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 6: Compare With/Without PCA")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(X_blob, y_blob, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Without PCA
clf_no_pca = SVC(kernel='rbf')
clf_no_pca.fit(X_train_scaled, y_train)
acc_no_pca = clf_no_pca.score(X_test_scaled, y_test)

# With PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

clf_pca = SVC(kernel='rbf')
clf_pca.fit(X_train_pca, y_train)
acc_pca = clf_pca.score(X_test_pca, y_test)

print(f"Original features: {X_train.shape[1]}")
print(f"PCA components: {pca.n_components_}")
print(f"\nAccuracy without PCA: {acc_no_pca:.4f}")
print(f"Accuracy with PCA:    {acc_pca:.4f}")

# ============================================================================
# EXAMPLE 7: Image Compression (Digits)
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 7: Image Compression (Digits)")
print("="*70)

digits = load_digits()
X_digits = digits.data

scaler_d = StandardScaler()
X_digits_scaled = scaler_d.fit_transform(X_digits)

pca_digits = PCA(n_components=0.99)
X_digits_compressed = pca_digits.fit_transform(X_digits_scaled)

print(f"Original: {X_digits.shape[1]} features (64 pixels)")
print(f"Compressed: {X_digits_compressed.shape[1]} features")
print(f"Compression: {(1 - X_digits_compressed.shape[1]/X_digits.shape[1])*100:.1f}%")
print(f"Variance retained: {sum(pca_digits.explained_variance_ratio_)*100:.2f}%")

# Reconstruct
X_reconstructed = pca_digits.inverse_transform(X_digits_compressed)
error = np.mean((X_digits - X_reconstructed)**2)
print(f"Reconstruction error (MSE): {error:.4f}")

# Show sample
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_digits[i].reshape(8, 8), cmap='gray')
    ax.axis('off')
plt.suptitle('Original Digits')
plt.savefig('digits_original.png', dpi=100)
plt.close()

# ============================================================================
# EXAMPLE 8: Explained Variance Plot
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 8: Explained Variance Analysis")
print("="*70)

pca_full = PCA()
pca_full.fit(X_scaled)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
bars = plt.bar(range(1, 5), pca_full.explained_variance_ratio_, color='steelblue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance per Component')
plt.xticks(range(1, 5))

plt.subplot(1, 2, 2)
plt.plot(range(1, 5), np.cumsum(pca_full.explained_variance_ratio_), 'bo-', linewidth=2)
plt.fill_between(range(1, 5), np.cumsum(pca_full.explained_variance_ratio_), alpha=0.3)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance')
plt.legend()
plt.xticks(range(1, 5))

plt.tight_layout()
plt.savefig('pca_variance.png', dpi=100)
plt.close()
print("Saved: pca_variance.png")

print("\nVariance summary:")
for i in range(4):
    print(f"   PC{i+1}: {pca_full.explained_variance_ratio_[i]*100:.2f}%")

# ============================================================================
# EXAMPLE 9: PCA Components as Linear Combinations
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 9: PCA Components as Linear Combinations")
print("="*70)

print("\nPrincipal Components (eigenvectors):")
feature_names = iris.feature_names
for i in range(4):
    print(f"\nPC{i+1}:")
    for j, (name, val) in enumerate(zip(feature_names, pca_full.components_[i])):
        print(f"   {name}: {val:+.4f}")

print("\nInterpretation:")
print("   PC1 = sepal_length*0.50 + sepal_width*-0.32 + petal_length*0.58 + petal_width*0.56")
print("   (Mainly sepal length and petal dimensions)")

# ============================================================================
# EXAMPLE 10: Inverse Transform (Reconstruction)
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 10: Inverse Transform & Reconstruction")
print("="*70)

X_sample = X_scaled[:5]

pca_recon = PCA(n_components=2)
X_pca_sample = pca_recon.fit_transform(X_sample)
X_reconstructed = pca_recon.inverse_transform(X_pca_sample)

print(f"Original shape: {X_sample.shape}")
print(f"After PCA: {X_pca_sample.shape}")
print(f"Reconstructed: {X_reconstructed.shape}")

print(f"\nVariance retained: {sum(pca_recon.explained_variance_ratio_)*100:.2f}%")

print("\nReconstruction error (first sample):")
print(f"   Original:  {X_sample[0][:4]}")
print(f"   Reconstructed: {X_reconstructed[0][:4]}")

error = np.linalg.norm(X_sample - X_reconstructed)
print(f"   Total error: {error:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("                           SUMMARY")
print("="*70)
print("""
PCA Key Points:
──────────────
• Finds orthogonal directions of maximum variance
• Reduces dimensionality while retaining most information
• Standardize data before applying PCA (important!)
• Use n_components=0.95 to retain 95% variance
• Use inverse_transform to reconstruct approximation

PCA Pipeline:
─────────────
1. StandardScaler (center & scale)
2. PCA (reduce dimensions)
3. ML algorithm (train)

Common Uses:
────────────
• Dimensionality reduction
• Visualization (2D/3D)
• Noise reduction
• Preprocessing for other models
• Feature extraction
""")

print("="*70)
print("                   EXAMPLES COMPLETE!")
print("="*70)