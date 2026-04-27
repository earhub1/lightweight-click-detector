import argparse
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
K = None  # será definido via argumento --k
RANDOM_STATE = 42
OUTPUT_PNG = os.path.join(os.path.dirname(__file__), "chains_kmeans_pca.png")

FEATURES = [
    # Temporal / cadência
    "n_clicks",
    "duration_sec",
    "click_rate_hz",
    # ICI
    "mean_ici_ms",
    "std_ici_ms",
    "cv_ici",
    "median_ici_ms",
    "min_ici_ms",
    "max_ici_ms",
    "p25_ici_ms",
    "p75_ici_ms",
    "ici_trend_slope",
    # Intensidade (score Z)
    "mean_peak_z",
    "std_peak_z",
    "cv_peak_z",
    "min_peak_z",
    "max_peak_z",
    # Frequência dominante
    "mean_dominant_freq_hz",
    "std_dominant_freq_hz",
    "cv_dominant_freq_hz",
    "min_dominant_freq_hz",
    "max_dominant_freq_hz",
    # Amplitude espectral
    "mean_dominant_amp",
    "std_dominant_amp",
    # Largura de banda
    "mean_bw_3db_hz",
    "std_bw_3db_hz",
    "mean_bw_10db_hz",
    "std_bw_10db_hz",
    # Centróide espectral
    "spectral_centroid_hz",
    # Energia relativa por banda (perfil espectral)
    "mean_band_energy_rel_1",
    "mean_band_energy_rel_2",
    "mean_band_energy_rel_3",
    "mean_band_energy_rel_4",
    "mean_band_energy_rel_5",
    "mean_band_energy_rel_6",
    "mean_band_energy_rel_7",
    "mean_band_energy_rel_8",
    "mean_band_energy_rel_9",
    "mean_band_energy_rel_10",
]

# Colunas com possíveis valores extremos — tratadas com clipping p99 + log1p
LOG_COLS = ["mean_peak_z", "std_peak_z", "cv_peak_z", "min_peak_z", "max_peak_z",
            "mean_dominant_amp", "std_dominant_amp"]
CLIP_COLS = LOG_COLS  # mesmas colunas recebem clipping antes do log

# ------------------------------------------------------------------
# Argumento de linha de comando
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="KMeans + PCA 2D sobre all_chains.csv")
parser.add_argument("csv_path", help="Caminho para o arquivo all_chains.csv")
parser.add_argument("--k", type=int, default=4, help="Número de clusters do KMeans (padrão: 4)")
args = parser.parse_args()
K = args.k

# ------------------------------------------------------------------
# Tarefa 1 — Carregar e pré-processar
# ------------------------------------------------------------------
df = pd.read_csv(args.csv_path)

# Manter apenas colunas que existem no CSV (ignora colunas ausentes silenciosamente)
available = [c for c in FEATURES if c in df.columns]
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    print(f"Aviso: {len(missing)} coluna(s) ausentes no CSV e ignoradas:\n  {missing}")

df_feat = df[available].copy()
df_feat = df_feat.dropna()
print(f"\nAmostras após remoção de NaN: {len(df_feat)}")

for col in CLIP_COLS:
    if col in df_feat.columns:
        threshold = df_feat[col].quantile(0.99)
        df_feat[col] = df_feat[col].clip(upper=threshold)

for col in LOG_COLS:
    if col in df_feat.columns:
        df_feat[col] = np.log1p(df_feat[col].clip(lower=0))

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_feat)

# ------------------------------------------------------------------
# Tarefa 2 — KMeans
# ------------------------------------------------------------------
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
labels = kmeans.fit_predict(X_scaled)
centroids_scaled = kmeans.cluster_centers_

centroids_original = scaler.inverse_transform(centroids_scaled)
for col in LOG_COLS:
    if col in available:
        idx = available.index(col)
        centroids_original[:, idx] = np.expm1(centroids_original[:, idx])

centroids_df = pd.DataFrame(centroids_original, columns=available)
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
print(f"\n  Features usadas: {len(available)}")
