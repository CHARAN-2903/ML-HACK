# Install necessary libraries
!pip install pandas rdkit-pypi scikit-learn matplotlib seaborn

# Import libraries
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Load the dataset
file_path = 'path_to_your_dataset.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Preprocess the molecular structures
def preprocess_molecules(smiles_list):
    mol_objects = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fingerprint_matrix = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mol_objects]
    fingerprint_array = [DataStructs.BitVectToNumpyArray(fp) for fp in fingerprint_matrix]
    return fingerprint_array

# Extract molecular structures and preprocess them
molecular_structures = data['SMILES'].tolist()
fingerprints = preprocess_molecules(molecular_structures)

# Apply PCA to reduce dimensionality (optional)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(fingerprints)

# Apply K-means clustering
num_clusters_kmeans = 5  # Adjust based on your requirements
kmeans = KMeans(n_clusters=num_clusters_kmeans, random_state=42)
cluster_labels_kmeans = kmeans.fit_predict(fingerprints)

# Apply hierarchical clustering
num_clusters_hierarchical = 5  # Adjust based on your requirements
hierarchical = AgglomerativeClustering(n_clusters=num_clusters_hierarchical, linkage='ward')
cluster_labels_hierarchical = hierarchical.fit_predict(fingerprints)

# Visualize the K-means results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels_kmeans, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='red', label='Centroids')
plt.title('K-Means Clustering of Small Molecules')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Visualize the hierarchical clustering results
plt.subplot(1, 2, 2)
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels_hierarchical, cmap='viridis', alpha=0.7)
plt.title('Hierarchical Clustering of Small Molecules')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

# Silhouette analysis for K-means
silhouette_avg_kmeans = silhouette_score(fingerprints, cluster_labels_kmeans)
print(f"Average silhouette score for K-means: {silhouette_avg_kmeans}")

# Silhouette analysis for hierarchical clustering
silhouette_avg_hierarchical = silhouette_score(fingerprints, cluster_labels_hierarchical)
print(f"Average silhouette score for Hierarchical Clustering: {silhouette_avg_hierarchical}")

# Compute silhouette scores for each sample in K-means
sample_silhouette_values_kmeans = silhouette_samples(fingerprints, cluster_labels_kmeans)

# Compute silhouette scores for each sample in hierarchical clustering
sample_silhouette_values_hierarchical = silhouette_samples(fingerprints, cluster_labels_hierarchical)

# Create subplots with 2 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Silhouette plots for K-means
y_lower_kmeans = 10
for i in range(num_clusters_kmeans):
    ith_cluster_silhouette_values_kmeans = sample_silhouette_values_kmeans[cluster_labels_kmeans == i]
    ith_cluster_silhouette_values_kmeans.sort()

    size_cluster_i_kmeans = ith_cluster_silhouette_values_kmeans.shape[0]
    y_upper_kmeans = y_lower_kmeans + size_cluster_i_kmeans

    color_kmeans = plt.cm.viridis(float(i) / num_clusters_kmeans)
    axs[0, 0].fill_betweenx(np.arange(y_lower_kmeans, y_upper_kmeans), 0, ith_cluster_silhouette_values_kmeans,
                            facecolor=color_kmeans, edgecolor=color_kmeans, alpha=0.7)

    axs[0, 0].text(-0.05, y_lower_kmeans + 0.5 * size_cluster_i_kmeans, str(i))

    y_lower_kmeans = y_upper_kmeans + 10

axs[0, 0].set_title("Silhouette plot for K-Means Clustering")
axs[0, 0].set_xlabel("Silhouette Coefficient Values")
axs[0, 0].set_ylabel("Cluster Label")
axs[0, 0].set_yticks([])  # Clear the y-axis labels/ticks
axs[0, 0].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# Silhouette plots for hierarchical clustering
y_lower_hierarchical = 10
for i in range(num_clusters_hierarchical):
    ith_cluster_silhouette_values_hierarchical = sample_silhouette_values_hierarchical[cluster_labels_hierarchical == i]
    ith_cluster_silhouette_values_hierarchical.sort()

    size_cluster_i_hierarchical = ith_cluster_silhouette_values_hierarchical.shape[0]
    y_upper_hierarchical = y_lower_hierarchical + size_cluster_i_hierarchical

    color_hierarchical = plt.cm.viridis(float(i) / num_clusters_hierarchical)
    axs[0, 1].fill_betweenx(np.arange(y_lower_hierarchical, y_upper_hierarchical),
                            0, ith_cluster_sil
