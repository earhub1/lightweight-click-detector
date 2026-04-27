import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import hdbscan
from sklearn.preprocessing import RobustScaler

# ------------------------------------------------------------------
# Configuração — features (idêntico ao analyze_chains.py)
# ------------------------------------------------------------------
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

LOG_COLS = ["mean_peak_z", "std_peak_z", "cv_peak_z", "min_peak_z", "max_peak_z",
            "mean_dominant_amp", "std_dominant_amp"]
CLIP_COLS = LOG_COLS

OUTPUT_PNG = os.path.join(os.path.dirname(__file__), "chains_umap_hdbscan.png")
RANDOM_STATE = 42

# ------------------------------------------------------------------
# Tarefa 1 — Argumentos de linha de comando
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="UMAP + HDBSCAN sobre all_chains.csv")
parser.add_argument("csv_path", help="Caminho para o arquivo all_chains.csv")
parser.add_argument("--n_neighbors", type=int, default=15,
                    help="Parâmetro UMAP: vizinhos locais (padrão: 15)")
parser.add_argument("--min_dist", type=float, default=0.1,
                    help="Parâmetro UMAP: distância mínima entre pontos (padrão: 0.1)")
parser.add_argument("--min_cluster_size", type=int, default=5,
                    help="Parâmetro HDBSCAN: tamanho mínimo de cluster (padrão: 5)")
args = parser.parse_args()

# ------------------------------------------------------------------
# Tarefa 2 — Carregar e pré-processar
# ------------------------------------------------------------------
df = pd.read_csv(args.csv_path)

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
# Tarefa 3 — UMAP 2D
# ------------------------------------------------------------------
print(f"\nRodando UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=args.n_neighbors,
    min_dist=args.min_dist,
    random_state=RANDOM_STATE,
)
X_umap = reducer.fit_transform(X_scaled)

# ------------------------------------------------------------------
# Tarefa 4 — HDBSCAN
# ------------------------------------------------------------------
print(f"Rodando HDBSCAN (min_cluster_size={args.min_cluster_size})...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size)
labels = clusterer.fit_predict(X_umap)

unique_clusters = sorted(set(labels[labels >= 0]))
n_clusters = len(unique_clusters)
noise_mask = labels == -1
n_noise = noise_mask.sum()
pct_noise = 100 * n_noise / len(labels)

# ------------------------------------------------------------------
# Tarefa 5 — Visualização
# ------------------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))

fig, ax = plt.subplots(figsize=(9, 7))

if n_noise > 0:
    ax.scatter(
        X_umap[noise_mask, 0],
        X_umap[noise_mask, 1],
        s=12,
        alpha=0.4,
        color="#cccccc",
        linewidths=0,
    )

for i, k in enumerate(unique_clusters):
    mask = labels == k
    ax.scatter(
        X_umap[mask, 0],
        X_umap[mask, 1],
        s=18,
        alpha=0.6,
        color=colors[i],
        label=f"Cluster {k} (n={mask.sum()})",
        linewidths=0,
    )

if n_noise > 0:
    ax.scatter([], [], s=12, color="#cccccc", alpha=0.6,
               label=f"Ruído (n={n_noise}, {pct_noise:.1f}%)")

ax.set_xlabel("UMAP1", fontsize=12)
ax.set_ylabel("UMAP2", fontsize=12)
ax.set_title(
    f"UMAP + HDBSCAN  |  n_neighbors={args.n_neighbors}  "
    f"min_dist={args.min_dist}  min_cluster_size={args.min_cluster_size}",
    fontsize=11,
)
ax.legend(loc="best", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"\nFigura salva em: {OUTPUT_PNG}")
plt.show()

# ------------------------------------------------------------------
# Tarefa 6 — Output no terminal
# ------------------------------------------------------------------
print(f"\n=== Resultado UMAP + HDBSCAN ===")
print(f"  Total de amostras: {len(labels)}")
print(f"  Clusters encontrados: {n_clusters}")

if n_clusters == 0:
    print("\n  AVISO: nenhum cluster encontrado — todos os pontos foram classificados como ruído.")
    print("  Sugestão: reduza --min_cluster_size (ex: --min_cluster_size 3)")
else:
    print("\n  Tamanho por cluster:")
    for k in unique_clusters:
        print(f"    Cluster {k}: {(labels == k).sum()} chains")

print(f"\n  Ruído: {n_noise} amostras ({pct_noise:.1f}%)")
print(f"  Features usadas: {len(available)}")
