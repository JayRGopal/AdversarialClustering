import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages


def compute_clustering_accuracy(embeddings, labels):
    """
    Computes the clustering accuracy given image embeddings and their corresponding labels.
    
    Args:
    - embeddings: numpy array of shape (n_samples, n_latent_dims)
    - labels: numpy array of shape (n_samples,)
    
    Returns:
    - accuracy: float value indicating clustering accuracy
    """
    # Define the number of clusters (in this case, the number of classes)
    n_clusters = len(np.unique(labels))
    
    # Cluster the embeddings using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    predicted_labels = kmeans.fit_predict(embeddings)
    
    # Compute clustering accuracy
    accuracy = np.sum(predicted_labels == labels) / len(labels)
    
    return accuracy




def save_clustering_plot(embeddings, labels, filepath):
    """
    Plots clustering on a random subset of classes
    
    Args:
    - embeddings: numpy array of shape (n_samples, n_latent_dims)
    - labels: numpy array of shape (n_samples,)
    - filepath: the place we want to save the pdf w the plot
    
    Returns: None
    - 
    """

    # Randomly select 5 classes
    random_classes = np.random.choice(np.unique(labels), size=5, replace=False)
    
    # Get the embeddings and labels for the selected classes
    mask = np.isin(labels, random_classes)
    selected_embeddings = embeddings[mask]
    selected_labels = labels[mask]
    
    # Perform PCA to reduce the embeddings to 2 dimensions
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(selected_embeddings)
    
    # Plot the clusters on a 2D plot
    with PdfPages(filepath) as pdf:
        plt.figure()
        for label in random_classes:
            mask = selected_labels == label
            plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], label=label, alpha=0.5)
        plt.legend()
        pdf.savefig()
    
    return

