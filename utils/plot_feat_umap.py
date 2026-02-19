import glob
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
from scipy.spatial.distance import cdist

import cudf
import pandas as pd
from cuml.cluster import HDBSCAN

from cuml import PCA
from cuml import UMAP, TSNE


plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16,
})

class Config:
    def paths():
        root = "./logs"
        file = sorted(glob.glob(f"{root}/*.parquet"))
        output = "./plots"

        return file, output
    
    def ml_models():

        dbscan = HDBSCAN(
            min_cluster_size=5,  
            min_samples=1,            
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom',  
            metric='euclidean',
            prediction_data=True
        )

        reducer = UMAP(
            n_components=2,
            n_neighbors=30,           # Critical parameter! (default: 15)
            min_dist=0.6,              # Critical parameter! (default: 0.1)
            metric='euclidean',        # Distance metric
            n_epochs=500,              # More epochs for better convergence (default: 200)
            learning_rate=1.0,         # Default is better than 1000 for UMAP
            init='spectral',           # Initialization method
            spread=5.0,                # Effective scale of embedded points
            negative_sample_rate=5,    # Increase for high-dimensional data
            random_state=42,
            verbose=True
        )

        return dbscan, reducer

file, output = Config.paths()

# prithvi = epochs true_label_mean  pred_label_mean  true_flood_pixels  total_pixels  flood_ratio     f_000     f_001     f_002     f_003     f_004     f_005     f_006     f_007     f_008     f_009     f_010     f_011
# prithvi_unet = epochs true_label_mean  pred_label_mean  true_flood_pixels  total_pixels  flood_ratio     f_000     f_001     f_002     f_003     f_004     f_005
# unet = epochs true_label_mean  pred_label_mean  true_flood_pixels  total_pixels  flood_ratio     f_000     f_001     f_002     f_003     f_004     f_005

prithvi_file = file[0]
prithvi_unet_file = file[1]
unet_file = file[2]

logger.info(f"Prithvi file: {prithvi_file}")

df_prithvi = cudf.read_parquet(f"{prithvi_file}")
df_prithvi_unet = cudf.read_parquet(f"{prithvi_unet_file}")
df_unet = cudf.read_parquet(f"{unet_file}")
 

prithvi_features = [col for col in df_prithvi.columns if col.startswith('f_')]
prithvi_unet_features = [col for col in df_prithvi_unet.columns if col.startswith('f_')]
unet_features = [col for col in df_unet.columns if col.startswith('f_')]


def cluster_dataframe(df, feature_columns):
    X = df[feature_columns]     
    dbscan, _ = Config.ml_models()
    labels = dbscan.fit_predict(X)
    df = df.copy()
    df['cluster'] = labels
    n_clusters = len(df['cluster'].unique())
    logger.info(f"Found {n_clusters} clusters (including noise as -1)")
    
    return df

df_prithvi_clustered = cluster_dataframe(df_prithvi, prithvi_features)
df_prithvi_unet_clustered = cluster_dataframe(df_prithvi_unet, prithvi_unet_features)
df_unet_clustered = cluster_dataframe(df_unet, unet_features)

 

def visualize_clusters2D(df, feature_columns, title="Clusters by Dominant Feature", save_path=None):
  
    X = df[feature_columns]
    _, reducer = Config.ml_models()
    X_np = X.to_numpy()
    X_2d = reducer.fit_transform(X_np)
    X_2d_pd = pd.DataFrame(X_2d)
     
    X_pd = X.to_pandas()
    dominant_feature_idx = X_pd.idxmax(axis=1)
     
    feature_names = feature_columns
     
    n_features = len(feature_names)
    cmap = plt.cm.hsv(np.linspace(0, 1, n_features))
    
    plt.figure(figsize=(12, 12))
     
    for i, feature in enumerate(feature_names):
        mask = dominant_feature_idx == feature
        if mask.any():
            feature = f"$f({i})$"
            plt.scatter(X_2d_pd.loc[mask, 0], X_2d_pd.loc[mask, 1],
                       c=[cmap[i]], alpha=0.2, s=26, label=f"${feature}$", edgecolors='none')
     
    epochs = df['epoch'].to_pandas()
    sorted_idx = np.argsort(epochs.values)
    sorted_points = X_2d_pd.iloc[sorted_idx].values
    sorted_epochs = epochs.iloc[sorted_idx].values

    unique_epochs = np.unique(sorted_epochs)
    
    centroids = []
    for epoch in unique_epochs:
        mask = sorted_epochs == epoch
        centroid = np.mean(sorted_points[mask], axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
 
    for i in range(len(centroids) - 1):
        plt.plot([centroids[i, 0], centroids[i+1, 0]], 
                [centroids[i, 1], centroids[i+1, 1]], 
                color='black',
                linewidth=4,
                alpha=1.0,
                zorder=5)
         
        # plt.scatter(centroids[i, 0], centroids[i, 1],
        #            marker='o', s=200, color='white', edgecolor='black',
        #            linewidth=1.5, zorder=6)
        if i == 0:
            plt.scatter(centroids[i, 0], centroids[i, 1],
                    marker='o', s=50, color='white', edgecolor='black',
                    linewidth=1.5, zorder=6, label='$\mu(Centroids)$')
        else:
            plt.scatter(centroids[i, 0], centroids[i, 1],
                    marker='o', s=50, color='white', edgecolor='black',
                    linewidth=1.5, zorder=6)
     
    if len(unique_epochs) > 0:
        start_mask = sorted_epochs == unique_epochs[0]
        start_centroid = sorted_points[start_mask].mean(axis=0)
        plt.scatter(start_centroid[0], start_centroid[1],
                   marker='^', s=250, color='white', edgecolor='black',
                   linewidth=1.5, zorder=10, label=f'Start E{int(unique_epochs[0])}')
        
        end_mask = sorted_epochs == unique_epochs[-1]
        end_centroid = sorted_points[end_mask].mean(axis=0)
        plt.scatter(end_centroid[0], end_centroid[1],
                   marker='*', s=600, color='gold', edgecolor='black',
                   linewidth=1.5, zorder=10, label=f'End E{int(unique_epochs[-1])}')
    
    plt.title(f"{title}")
    plt.xlabel("D1")
    plt.ylabel("D2")
    plt.legend(fontsize=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return X_2d




umap_prithvi = visualize_clusters2D(df_prithvi_clustered, prithvi_features, 
                                  title="Prithvi", 
                                  save_path=f"{output}/prithvi_clusters.png")

umap_prithvi_unet = visualize_clusters2D(df_prithvi_unet_clustered, prithvi_unet_features,
                                       title="Prithvi-UNet",
                                       save_path=f"{output}/prithvi_unet_clusters.png")

umap_unet = visualize_clusters2D(df_unet_clustered, unet_features,
                               title="UNet",
                               save_path=f"{output}/unet_clusters.png")