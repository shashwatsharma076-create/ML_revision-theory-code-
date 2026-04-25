"""
================================================================================
                    CLUSTERING - Complete Implementation
================================================================================

This module covers:
1. K-Means Clustering
2. Hierarchical Clustering
3. DBSCAN
4. Gaussian Mixture Models
5. Evaluation
6. Practical Examples

Author: ML Revision Series
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.datasets import (
    make_blobs,
    make_moons,
    make_circles,
    make_classification
)
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# PART 1: K-MEANS CLUSTERING
# =============================================================================

def example_kmeans():
    """
    Example 1: K-Means Clustering
    """
    print("=" * 70)
    print("EXAMPLE 1: K-Means Clustering")
    print("=" * 70)
    
    # Generate data
    X, y = make_blobs(n_samples=300, centers=3, random_state=42)
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test different K values
    print("\n📊 K-Means Results:")
    print("-" * 50)
    
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        sil = silhouette_score(X, labels)
        wcss = kmeans.inertia_
        
        print(f"   K={k}: Silhouette={sil:.4f}, WCSS={int(wcss)}")
    
    # Show centroids
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    print(f"\n📍 Centroids (first 3):")
    for i, centroid in enumerate(kmeans.cluster_centers_):
        print(f"   Cluster {i}: {centroid[:2]}")


# =============================================================================
# PART 2: ELBOW METHOD
# =============================================================================

def example_elbow_method():
    """
    Example 2: Finding Optimal K using Elbow Method
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Elbow Method")
    print("="*70)
    
    X, y = make_blobs(n_samples=500, centers=3, random_state=42)
    
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    print("\n📊 WCSS for each K:")
    print("-" * 40)
    for k, w in enumerate(wcss, 1):
        bar = "█" * int(w / 5000)
        print(f"   K={k:<2}: {bar} {int(w)}")
    
    print(f"\n📈 Elbow at approximately K=3")


# =============================================================================
# PART 3: HIERARCHICAL CLUSTERING
# =============================================================================

def example_hierarchical():
    """
    Example 3: Hierarchical Clustering
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Hierarchical Clustering")
    print("="*70)
    
    X, y = make_blobs(n_samples=200, centers=3, random_state=42)
    
    link_methods = ['ward', 'complete', 'average', 'single']
    
    print("\n📊 Linkage Methods:")
    print("-" * 50)
    
    for method in link_methods:
        agg = AgglomerativeClustering(n_clusters=3, linkage=method)
        labels = agg.fit_predict(X)
        
        sil = silhouette_score(X, labels)
        
        print(f"   {method:<10}: Silhouette={sil:.4f}")


# =============================================================================
# PART 4: DBSCAN
# =============================================================================

def example_dbscan():
    """
    Example 4: DBSCAN
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: DBSCAN")
    print("="*70)
    
    # Moon data - not spherical clusters
    X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
    
    for eps in [0.1, 0.2, 0.3, 0.5]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"   eps={eps}: {n_clusters} clusters, {n_noise} noise")


# =============================================================================
# PART 5: GAUSSIAN MIXTURE MODELS
# =============================================================================

def example_gmm():
    """
    Example 5: Gaussian Mixture Models
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Gaussian Mixture Models")
    print("="*70)
    
    X, y = make_blobs(n_samples=300, centers=3, random_state=42)
    
    for n in [2, 3, 4, 5]:
        gmm = GaussianMixture(n_components=n, random_state=42)
        labels = gmm.fit_predict(X)
        
        sil = silhouette_score(X, labels)
        
        print(f"   n={n}: Silhouette={sil:.4f}")
    
    # Show soft assignments
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)
    
    probs = gmm.predict_proba(X[:5])
    print(f"\n📊 First 5 points probabilities:")
    print(probs)


# =============================================================================
# PART 6: COMPARE ALL METHODS
# =============================================================================

def example_compare():
    """
    Example 6: Compare All Clustering Methods
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Compare All Methods")
    print("="*70)
    
    X, y = make_blobs(n_samples=300, centers=3, random_state=42)
    
    methods = {
        'K-Means': KMeans(n_clusters=3, random_state=42),
        'Hierarchical': AgglomerativeClustering(n_clusters=3),
        'GMM': GaussianMixture(n_components=3, random_state=42)
    }
    
    print("\n📊 Comparison:")
    print("-" * 50)
    
    for name, model in methods.items():
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        print(f"   {name:<15}: Silhouette={sil:.4f}")


# =============================================================================
# PART 7: CUSTOMER SEGMENTATION
# =============================================================================

def example_customer_segmentation():
    """
    Example 7: Customer Segmentation
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Customer Segmentation")
    print("="*70)
    
    np.random.seed(42)
    n = 500
    
    # Generate customer data
    income = np.random.normal(50000, 15000, n)
    spending = np.random.normal(50, 15, n)
    age = np.random.normal(35, 10, n)
    
    X = np.column_stack([income, spending, age])
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    print("\n📊 Customer Segments:")
    print("-" * 60)
    
    for i in range(5):
        mask = labels == i
        print(f"\n   Segment {i}: {mask.sum()} customers")
        print(f"      Avg Income: ${income[mask].mean():.0f}")
        print(f"      Avg Spending Score: {spending[mask].mean():.0f}")
        print(f"      Avg Age: {age[mask].mean():.0f}")


# =============================================================================
# PART 8: SILHOUETTE EVALUATION
# =============================================================================

def example_silhouette():
    """
    Example 8: Silhouette Evaluation
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Silhouette Scores")
    print("="*70)
    
    X, y = make_blobs(n_samples=300, centers=3, random_state=42)
    
    # Different cluster counts
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        sil = silhouette_score(X, labels)
        dav = davies_bouldin_score(X, labels)
        cal = calinski_harabasz_score(X, labels)
        
        print(f"\n   K={k}:")
        print(f"      Silhouette:    {sil:.4f} (higher is better)")
        print(f"      Davies-Bouldin: {dav:.4f} (lower is better)")
        print(f"      Calinski-Harabasz: {int(cal)} (higher is better)")


# =============================================================================
# PART 9: ANOMALY DETECTION
# =============================================================================

def example_anomaly_detection():
    """
    Example 9: Anomaly Detection with DBSCAN
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: Anomaly Detection")
    print("="*70)
    
    # Generate data with outliers
    X, y = make_blobs(n_samples=200, centers=2, random_state=42)
    
    # Add outliers
    outliers = np.random.uniform(-15, 15, (20, 2))
    X_outliers = np.vstack([X, outliers])
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_outliers)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\n📊 Results:")
    print(f"   Normal clusters: {n_clusters}")
    print(f"   Anomalies detected: {n_noise}")
    print(f"   Normal points: {len(labels) - n_noise}")


# =============================================================================
# PART 10: NON-CONVEX CLUSTERS
# =============================================================================

def example_non_convex():
    """
    Example 10: Non-Convex Clusters
    """
    print("\n" + "="*70)
    print("EXAMPLE 10: Non-Convex Clusters")
    print("="*70)
    
    # Moon data - non-convex
    X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
    
    print("\n📊 Cluster methods on moon data:")
    print("-" * 50)
    
    # K-Means fails on non-convex
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_sil = silhouette_score(X, kmeans_labels)
    print(f"   K-Means: Silhouette={kmeans_sil:.4f} (assumes spherical)")
    
    # DBSCAN works better
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    if n_clusters == 2:
        dbscan_sil = silhouette_score(X, dbscan_labels)
        print(f"   DBSCAN: Silhouette={dbscan_sil:.4f} (detects shape)")
    
    # GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(X)
    gmm_sil = silhouette_score(X, gmm_labels)
    print(f"   GMM: Silhouette={gmm_sil:.4f} (soft clustering)")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary():
    """Print clustering summary."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                       CLUSTERING SUMMARY                                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   TYPES:                                                             ║
║   • K-Means: Centroid-based, requires K                              ║
║   • Hierarchical: Tree structure, no K needed                         ║
║   • DBSCAN: Density-based, finds outliers, no K needed              ║
║   • GMM: Probabilistic, soft assignments                                ║
║                                                                       ║
║   EVALUATION:                                                        ║
║   • Silhouette: -1 to 1 (higher is better)                            ║
║   • Davies-Bouldin: Lower is better                                 ║
║   • Calinski-Harabasz: Higher is better                              ║
║                                                                       ║
║   PROS:                                                              ║
║   ✓ No labels needed                                                ║
║   ✓ Finds hidden patterns                                            ║
║   ✓ Good for exploration                                             ║
║                                                                       ║
║   CONS:                                                              ║
║   ✗ Hard to evaluate (no ground truth)                             ║
║   ✗ K-Means sensitive to initialization                            ║
║   ✗ Assumes cluster shapes                                          ║
║                                                                       ║
║   BEST FOR:                                                          ║
║   • Customer segmentation                                          ║
║   • Anomaly detection                                               ║
║   • Document organization                                          ║
║   • Image compression                                               ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║               CLUSTERING - Complete Implementation                     ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n" + "▶" * 35)
    print(" RUNNING ALL EXAMPLES")
    print("▶" * 35)
    
    # Run examples
    example_kmeans()
    example_elbow_method()
    example_hierarchical()
    example_dbscan()
    example_gmm()
    example_compare()
    example_customer_segmentation()
    example_silhouette()
    example_anomaly_detection()
    example_non_convex()
    
    # Summary
    print_summary()
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                        COMPLETED!                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# QUICK FUNCTIONS
# =============================================================================

def quick_kmeans(X, n_clusters=3):
    """Quick K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(X)


def quick_dbscan(X, eps=0.5, min_samples=5):
    """Quick DBSCAN clustering."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(X)


# =============================================================================
# END OF FILE
# =============================================================================