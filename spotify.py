# ==================== IMPORT LIBRARIES ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os

# ==================== LOAD DATA ====================
df = pd.read_csv("spotify dataset.csv")
print("Initial Data Shape:", df.shape)
print(df.head())

# ==================== HANDLE MISSING VALUES ====================
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# ==================== CREATE FOLDER FOR PLOTS ====================
os.makedirs("plots", exist_ok=True)

# ==================== EDA VISUALIZATIONS ====================
# Playlist Genre Distribution
plt.figure(figsize=(10,5))
sns.countplot(y='playlist_genre', data=df, order=df['playlist_genre'].value_counts().index)
plt.title("Playlist Genre Distribution")
plt.savefig("plots/playlist_genre_distribution.png")
plt.show()

# Playlist Name Distribution (Top 15)
top_playlists = df['playlist_name'].value_counts().head(15).index
plt.figure(figsize=(10,6))
sns.countplot(y='playlist_name', data=df[df['playlist_name'].isin(top_playlists)], 
              order=top_playlists)
plt.title("Top 15 Playlist Names")
plt.savefig("plots/top_playlist_names.png")
plt.show()

# Distribution of numeric audio features
numeric_cols = ['danceability','energy','key','loudness','mode','speechiness',
                'acousticness','instrumentalness','liveness','valence','tempo',
                'duration_ms','track_popularity']

df[numeric_cols].hist(figsize=(15,12), bins=20)
plt.tight_layout()
plt.savefig("plots/numeric_features_distribution.png")
plt.show()

# ==================== CORRELATION MATRIX ====================
plt.figure(figsize=(12,8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("plots/correlation_matrix.png")
plt.show()

# ==================== FEATURE SCALING ====================
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[numeric_cols])
scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)

# ==================== K-MEANS CLUSTERING ====================
# Determine optimal cluster count using elbow method
inertia = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'o-')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.savefig("plots/elbow_method.png")
plt.show()

# Choose cluster count (e.g., k=5)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)

# ==================== PCA FOR VISUALIZATION ====================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df['PCA1'] = pca_result[:,0]
df['PCA2'] = pca_result[:,1]

plt.figure(figsize=(10,6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='tab10')
plt.title("K-Means Clusters Visualization (PCA)")
plt.savefig("plots/cluster_visualization_pca.png")
plt.show()

# ==================== CLUSTERS BY PLAYLIST GENRE ====================
plt.figure(figsize=(10,6))
sns.countplot(x='playlist_genre', hue='Cluster', data=df)
plt.title("Clusters Grouped by Playlist Genre")
plt.savefig("plots/clusters_by_playlist_genre.png")
plt.show()

# ==================== SIMPLE RECOMMENDATION SYSTEM ====================
def recommend_song(song_name, n_recommendations=5):
    if song_name not in df['track_name'].values:
        return "Song not found!"
    idx = df[df['track_name'] == song_name].index[0]
    song_vector = scaled_features[idx].reshape(1, -1)
    similarities = cosine_similarity(song_vector, scaled_features).flatten()
    similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
    return df.iloc[similar_indices][['track_name', 'track_artist', 'playlist_genre', 'Cluster']]

# Example Recommendation
print("Recommendations for 'Shape of You' (if exists):")
print(recommend_song("Shape of You"))

# ==================== SAVE FINAL PROCESSED DATA ====================
df.to_csv("spotify_processed_with_clusters.csv", index=False)
print("All plots saved in 'plots/' folder and processed dataset saved as 'spotify_processed_with_clusters.csv'")
