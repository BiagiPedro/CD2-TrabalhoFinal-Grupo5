import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator   # pip install kneed



# =================================================================
# 🎨  ESTILO
# =================================================================
plt.rcParams.update({
    "figure.facecolor": "#0D1117", "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D", "axes.labelcolor":  "#C9D1D9",
    "axes.titlecolor":  "#E6EDF3", "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E", "text.color":       "#C9D1D9",
    "grid.color":       "#21262D", "grid.linestyle":   "--",
    "legend.facecolor": "#161B22", "legend.edgecolor": "#30363D",
    "figure.dpi": 130, "font.family": "monospace",
})

COR_C0      = "#3498DB"   # Cluster 0 — Guerra
COR_C1      = "#2ECC71"   # Cluster 1 — Paz
COR_OUT     = "#E74C3C"   # Outlier
COR_GUERRA  = "#F0A500"
COR_PAZ     = "#444C56"

# ── Terminal helpers ─────────────────────────────────────────────
def header(title, width=70, char="═"):
    print(f"\n\033[1;36m  {char*width}\033[0m")
    print(f"\033[1;97m    {title}\033[0m")
    print(f"\033[1;36m  {char*width}\033[0m")

def subheader(title, char="─", width=62):
    pad = max(width - len(title) - 5, 1)
    print(f"\n  \033[1;33m{char*3} {title} {char*pad}\033[0m")

def badge(label, value, color="97"):
    print(f"  \033[36m▸ {label:<34}\033[0m \033[{color}m{value}\033[0m")

# =================================================================
# 1️⃣  CARREGAMENTO
# =================================================================
path = "preprocessamento10anos/base10anosprocessada.csv"
df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)

num_cols = ["Valor US$ FOB", "Quilograma Líquido"]
bin_cols = ["Produto_Estrategico", "Fluxo", "Periodo_Guerra"]
cat_cols = ["UF do Produto"]
df = df.dropna(subset=num_cols + bin_cols + cat_cols).reset_index(drop=True)

# =================================================================
# 2️⃣  PRÉ-PROCESSAMENTO
# =================================================================
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ('bin', 'passthrough', bin_cols),
])
X = preprocessor.fit_transform(df)
n_features = X.shape[1]
n_samples  = X.shape[0]

# =================================================================
# 3️⃣  CÁLCULO AUTOMÁTICO DE min_samples
# =================================================================
# Regra padrão da literatura (Ester et al. 1996, Sander et al. 1998):
#   min_samples = 2 × n_features   para dados de alta dimensionalidade
#   mínimo absoluto recomendado: 5  (evita clusters triviais)
#   cap prático: nunca ultrapassar ~ln(n) para bases grandes

min_samples_formula  = 2 * n_features
min_samples_ln       = max(5, int(np.log(n_samples)))
min_samples_escolhido = max(5, min(min_samples_formula, min_samples_ln))

# =================================================================
# 4️⃣  CÁLCULO AUTOMÁTICO DE eps — GRÁFICO K-DISTANCE
# =================================================================
# O eps ideal é o joelho da curva de distâncias ao k-ésimo vizinho
# (onde k = min_samples). Abaixo do joelho = região densa (cluster);
# acima = ruído.

print(f"\n  Calculando k-distance para k={min_samples_escolhido}... aguarde.")
nbrs = NearestNeighbors(n_neighbors=min_samples_escolhido, n_jobs=-1)
nbrs.fit(X)
distances, _ = nbrs.kneighbors(X)
dist_k = np.sort(distances[:, min_samples_escolhido - 1])

kl = KneeLocator(
    range(len(dist_k)), dist_k,
    curve='convex', direction='increasing', interp_method='polynomial'
)
eps_joelho = float(dist_k[kl.knee]) if kl.knee else float(np.percentile(dist_k, 90))

# Também testamos ±20% para sensitivity check
eps_baixo = round(eps_joelho * 0.80, 4)
eps_alto  = round(eps_joelho * 1.20, 4)
eps_joelho = round(eps_joelho, 4)

# =================================================================
# 5️⃣  TERMINAL — Justificativa dos parâmetros
# =================================================================
header("CÁLCULO AUTOMÁTICO DOS PARÂMETROS DBSCAN")

subheader("Dimensões da matriz X")
badge("Amostras (n_samples)",  f"{n_samples:,}")
badge("Features (n_features)", f"{n_features}")

subheader("min_samples — derivação")
print(f"""
  Regra base  :  2 × n_features  = 2 × {n_features} = {min_samples_formula}
  Regra ln(n) :  ln({n_samples}) ≈ {np.log(n_samples):.1f}  →  {min_samples_ln}

  A literatura recomenda 2×n_features para dados de alta
  dimensionalidade (Ester et al., 1996). Para bases grandes,
  ln(n) serve como teto para evitar clusters excessivamente
  restritivos.

  Valor usado:  min_samples = min(2×d, ln(n)) = \033[1;92m{min_samples_escolhido}\033[0m

  Interpretação: um ponto só será núcleo de cluster se tiver
  ao menos {min_samples_escolhido} vizinhos dentro do raio eps. Isso garante
  que apenas regiões genuinamente densas formem grupos.
""")

subheader("eps — derivação pelo k-distance (joelho da curva)")
print(f"""
  Para cada ponto, calculou-se a distância ao seu {min_samples_escolhido}º
  vizinho mais próximo. Essas distâncias, ordenadas, formam
  uma curva:
    · Trecho plano inicial  → pontos em regiões densas (clusters)
    · Joelho (inflexão)     → fronteira densidade/ruído  ← eps
    · Trecho íngreme final  → pontos isolados (outliers)

  O KneeLocator (algoritmo de curvatura) detectou o joelho em:

    eps (joelho)  =  \033[1;92m{eps_joelho}\033[0m
    eps − 20%     =  {eps_baixo}   (mais restritivo → mais outliers)
    eps + 20%     =  {eps_alto}   (mais permissivo → menos outliers)

  Um eps muito pequeno fragmenta clusters reais em muitos grupos.
  Um eps muito grande funde clusters distintos em um só.
""")

# =================================================================
# 6️⃣  GRÁFICO 1 — k-distance com joelho marcado
# =================================================================
fig_kd, ax_kd = plt.subplots(figsize=(12, 5))
ax_kd.plot(range(len(dist_k)), dist_k, color="#58A6FF", lw=1.5, label="Distância ao k-ésimo vizinho")
ax_kd.axhline(eps_joelho, color=COR_GUERRA, lw=2, linestyle="--",
              label=f"eps (joelho) = {eps_joelho}")
ax_kd.axhline(eps_baixo, color="#8B949E", lw=1, linestyle=":",
              label=f"eps − 20% = {eps_baixo}")
ax_kd.axhline(eps_alto,  color="#8B949E", lw=1, linestyle=":",
              label=f"eps + 20% = {eps_alto}")
if kl.knee:
    ax_kd.axvline(kl.knee, color=COR_GUERRA, lw=1, linestyle=":", alpha=0.5)
    ax_kd.scatter([kl.knee], [eps_joelho], color=COR_GUERRA, s=120, zorder=5)
    ax_kd.annotate(f"  Joelho\n  eps={eps_joelho}",
                   xy=(kl.knee, eps_joelho),
                   xytext=(kl.knee + len(dist_k)*0.05, eps_joelho * 1.15),
                   arrowprops=dict(arrowstyle="->", color=COR_GUERRA),
                   color=COR_GUERRA, fontsize=9)
ax_kd.set_title(f"k-Distance Plot  (k = min_samples = {min_samples_escolhido})\n"
                f"O joelho indica o eps ideal para separar clusters de ruído",
                fontsize=11, pad=12)
ax_kd.set_xlabel("Pontos ordenados por distância")
ax_kd.set_ylabel(f"Distância ao {min_samples_escolhido}º vizinho")
ax_kd.legend(fontsize=9); ax_kd.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

