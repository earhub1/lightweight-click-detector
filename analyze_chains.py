import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

# ------------------------------------------------------------------
# Configuração
# ------------------------------------------------------------------
K = 4
RANDOM_STATE = 42
CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "detections", "all_chains.csv"
)
OUTPUT_PNG = os.path.join(os.path.dirname(__file__), "chains_kmeans_pca.png")

FEATURES = [
    "n_clicks",
    "duration_sec",
    "click_rate_hz",
    "mean_ici_ms",
    "std_ici_ms",
    "cv_ici",
    "median_ici_ms",
    "ici_trend_slope",
    "mean_peak_z",
    "cv_peak_z",
]

LOG_COLS = ["mean_peak_z", "cv_peak_z"]
CLIP_COL = "mean_peak_z"

# ------------------------------------------------------------------
# Tarefa 1 — Carregar e pré-processar
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

df_feat = df[FEATURES].copy()
df_feat = df_feat.dropna()

clip_threshold = df_feat[CLIP_COL].quantile(0.99)
df_feat[CLIP_COL] = df_feat[CLIP_COL].clip(upper=clip_threshold)

for col in LOG_COLS:
    df_feat[col] = np.log1p(df_feat[col])

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_feat)

# ------------------------------------------------------------------
# Tarefa 2 — KMeans
# ------------------------------------------------------------------
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
labels = kmeans.fit_predict(X_scaled)
centroids_scaled = kmeans.cluster_centers_

centroids_original = scaler.inverse_transform(centroids_scaled)
for col_idx, col in enumerate(LOG_COLS):
    feat_idx = FEATURES.index(col)
    centroids_original[:, feat_idx] = np.expm1(centroids_original[:, feat_idx])

centroids_df = pd.DataFrame(centroids_original, columns=FEATURES)
centroids_df.index.name = "cluster"

# ------------------------------------------------------------------
# Tarefa 3 — PCA 2D
# ------------------------------------------------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(centroids_scaled)

var_explained = pca.explained_variance_ratio_ * 100
var_cumulative = var_explained.sum()

# ------------------------------------------------------------------
# Tarefa 4 — Visualização
# ------------------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, K))

fig, ax = plt.subplots(figsize=(9, 7))

for k in range(K):
    mask = labels == k
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=18,
        alpha=0.55,
        color=colors[k],
        label=f"Cluster {k} (n={mask.sum()})",
        linewidths=0,
    )

for k in range(K):
    ax.scatter(
        centroids_pca[k, 0],
        centroids_pca[k, 1],
        s=250,
        marker="X",
        color=colors[k],
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
    )

ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)", fontsize=12)
ax.set_title(f"KMeans k={K} — PCA 2D", fontsize=14)
ax.legend(loc="best", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"\nFigura salva em: {OUTPUT_PNG}")
plt.show()

# ------------------------------------------------------------------
# Tarefa 5 — Output no terminal
# ------------------------------------------------------------------
print("\n=== Chains por cluster ===")
for k in range(K):
    print(f"  Cluster {k}: {(labels == k).sum()} chains")

print("\n=== Centroids (espaço original) ===")
print(centroids_df.T.to_string(float_format=lambda x: f"{x:.4g}"))

print(f"\n=== Variância explicada pelo PCA ===")
print(f"  PC1: {var_explained[0]:.1f}%")
print(f"  PC2: {var_explained[1]:.1f}%")
print(f"  Acumulada: {var_cumulative:.1f}%")
