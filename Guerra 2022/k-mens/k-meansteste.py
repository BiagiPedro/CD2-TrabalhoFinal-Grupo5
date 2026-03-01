import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator   # pip install kneed
from pathlib import Path

# =================================================================
# 🎨  ESTILO GLOBAL
# =================================================================
CORES_CLUSTER_PALETTE = ["#2ECC71", "#E74C3C", "#3498DB",
                          "#F39C12", "#9B59B6", "#1ABC9C"]

plt.rcParams.update({
    "figure.facecolor": "#0D1117", "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D", "axes.labelcolor":  "#C9D1D9",
    "axes.titlecolor":  "#E6EDF3", "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E", "text.color":       "#C9D1D9",
    "grid.color":       "#21262D", "grid.linestyle":   "--",
    "legend.facecolor": "#161B22", "legend.edgecolor": "#30363D",
    "figure.dpi": 130, "font.family": "monospace",
})

# ── Terminal helpers ─────────────────────────────────────────────
def header(title, width=72, char="═"):
    print(f"\n\033[1;36m  {char*width}\033[0m")
    print(f"\033[1;97m    {title}\033[0m")
    print(f"\033[1;36m  {char*width}\033[0m")

def subheader(title, char="─", width=64):
    pad = max(width - len(title) - 5, 1)
    print(f"\n  \033[1;33m{char*3} {title} {char*pad}\033[0m")

def bar_ascii(value, max_val=1.0, width=20, char="█", bg="░"):
    filled = int((value / max_val) * width) if max_val else 0
    return (f"\033[36m{char*filled}\033[90m{bg*(width-filled)}\033[0m  {value:.1%}")

def fmt_fob(v):
    if abs(v) >= 1e9: return f"US$ {v/1e9:.2f} Bi"
    if abs(v) >= 1e6: return f"US$ {v/1e6:.2f} Mi"
    if abs(v) >= 1e3: return f"US$ {v/1e3:.2f} K"
    return f"US$ {v:.2f}"

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

# =================================================================
# 3️⃣  VARREDURA  K = 2…10
# =================================================================
K_MIN, K_MAX = 2, 10
k_range = list(range(K_MIN, K_MAX + 1))

print(f"\n  Varrendo K = {K_MIN}…{K_MAX}, aguarde…")
results = {"k": [], "inertia": [], "silhouette": [], "davies_bouldin": []}

for k in k_range:
    km     = KMeans(n_clusters=k, init='k-means++', random_state=9, n_init=10)
    labels = km.fit_predict(X)
    results["k"].append(k)
    results["inertia"].append(km.inertia_)
    results["silhouette"].append(
        silhouette_score(X, labels, sample_size=10_000, random_state=42))
    results["davies_bouldin"].append(davies_bouldin_score(X, labels))
    print(f"    K={k:2d}  Sil={results['silhouette'][-1]:.4f}"
          f"  DB={results['davies_bouldin'][-1]:.4f}"
          f"  Inércia={results['inertia'][-1]:.0f}")

df_res = pd.DataFrame(results)

# =================================================================
# 4️⃣  K ÓTIMO POR CADA ÍNDICE
# =================================================================
k_sil = int(df_res.loc[df_res["silhouette"].idxmax(), "k"])
k_db  = int(df_res.loc[df_res["davies_bouldin"].idxmin(), "k"])

try:
    kl = KneeLocator(
        df_res["k"].tolist(), df_res["inertia"].tolist(),
        curve="convex", direction="decreasing", interp_method="polynomial",
    )
    k_elbow = int(kl.knee) if kl.knee else int(
        df_res.iloc[df_res["inertia"].diff().abs().iloc[1:].idxmax()]["k"])
except Exception:
    k_elbow = int(df_res.iloc[df_res["inertia"].diff().abs().iloc[1:].idxmax()]["k"])

votos_map = {
    "Silhouette Score": k_sil,
    "Davies-Bouldin":   k_db,
    "Elbow (WCSS)":     k_elbow,
}

# =================================================================
# 5️⃣  VOTAÇÃO MAJORITÁRIA
# =================================================================
from collections import Counter

contagem  = Counter(votos_map.values())
k_ideal   = contagem.most_common(1)[0][0]
max_votos = contagem[k_ideal]
tiebreak  = max_votos == 1
if tiebreak:
    k_ideal = k_sil

vencedores = [idx for idx, k in votos_map.items() if k == k_ideal]
row_k      = df_res[df_res["k"] == k_ideal].iloc[0]
sil_k      = row_k["silhouette"]
db_k       = row_k["davies_bouldin"]
ine_k      = row_k["inertia"]

def delta_inercia(k):
    sub = df_res.set_index("k")["inertia"]
    return sub[k-1] - sub[k] if (k-1 in sub.index and k in sub.index) else np.nan

queda_antes  = delta_inercia(k_elbow)
queda_depois = delta_inercia(k_elbow + 1) if k_elbow + 1 <= K_MAX else np.nan

# =================================================================
# 6️⃣  TERMINAL — Votação e Justificativa
# =================================================================
header("SELEÇÃO AUTOMÁTICA DO K  —  VOTAÇÃO MAJORITÁRIA (3 índices)")

subheader("Votos individuais de cada índice")
for idx, k_v in votos_map.items():
    check = "\033[92m✔\033[0m" if k_v == k_ideal else "\033[90m✘\033[0m"
    destq = "\033[92m" if k_v == k_ideal else "\033[90m"
    print(f"  {check}  \033[36m{idx:<22}\033[0m → {destq}K = {k_v}\033[0m")

print()
if tiebreak:
    print("  \033[93m⚠  Empate total — desempate pelo Silhouette Score.\033[0m")

print(f"\n  \033[1;97m{'─'*46}\033[0m")
print(f"  \033[1;92m  🏆  K IDEAL SELECIONADO: K = {k_ideal}  ({max_votos}/3 votos)\033[0m")
print(f"  \033[1;97m{'─'*46}\033[0m")

subheader("Justificativa Detalhada da Escolha")

sil_qualidade = (
    "boa coesão interna (> 0.50)"         if sil_k > 0.50 else
    "coesão moderada (entre 0.25 e 0.50)" if sil_k > 0.25 else
    "coesão fraca (< 0.25)"
)
sil_concordou = "concordou" if k_sil == k_ideal else f"divergiu (sugeriu K={k_sil})"

print(f"""
  📐 SILHOUETTE SCORE
  ──────────────────────────────────────────────────────────
  Valor em K={k_ideal}: {sil_k:.4f}
  Melhor K segundo este índice: {k_sil}  → {sil_concordou} com o resultado final.

  Mede o quão bem cada ponto se encaixa no seu cluster em
  relação ao cluster vizinho mais próximo. Varia de -1 a +1.
  Com K={k_ideal}, o índice aponta {sil_qualidade}.
""")

db_qualidade = (
    "clusters bem definidos e separados (< 1.0)"  if db_k < 1.0 else
    "separação moderada entre clusters (1.0–2.0)" if db_k < 2.0 else
    "alta sobreposição entre clusters (> 2.0)"
)
db_concordou = "concordou" if k_db == k_ideal else f"divergiu (sugeriu K={k_db})"

print(f"""  📏 DAVIES-BOULDIN INDEX
  ──────────────────────────────────────────────────────────
  Valor em K={k_ideal}: {db_k:.4f}
  Melhor K segundo este índice: {k_db}  → {db_concordou} com o resultado final.

  Calcula a razão entre dispersão interna e distância entre
  centróides. Quanto menor, mais separados os grupos.
  Com K={k_ideal}, o índice aponta {db_qualidade}.
""")

razao_txt = (
    f"{queda_antes/queda_depois:.1f}× maior que a queda K={k_elbow}→{k_elbow+1}"
    if not np.isnan(queda_depois) and queda_depois > 0
    else "queda posterior indisponível"
)
elbow_concordou = "concordou" if k_elbow == k_ideal else f"divergiu (sugeriu K={k_elbow})"

print(f"""  📉 ELBOW METHOD (Joelho da Curva WCSS)
  ──────────────────────────────────────────────────────────
  Joelho detectado em: K={k_elbow}  → {elbow_concordou} com o resultado final.
  Inércia em K={k_ideal}: {ine_k:,.0f}

  A queda de inércia K={k_elbow-1}→{k_elbow} foi {razao_txt}.
  {"Confirma K=" + str(k_elbow) + " como ponto de inflexão natural da curva."
   if k_elbow == k_ideal else
   "O joelho e a votação final divergem; decidido pelos demais índices."}
""")

print(f"""  ✅ CONCLUSÃO
  ──────────────────────────────────────────────────────────
  Por votação majoritária ({max_votos}/3{"  + desempate pelo Silhouette" if tiebreak else ""}),
  K = {k_ideal} foi selecionado como a partição ótima dos dados.

  {"Todos os índices convergem — escolha robusta." if max_votos == 3
   else "Dois de três índices concordam. O divergente aponta estrutura diferente,"}
  {"" if max_votos == 3
   else "  mas a maioria indica K=" + str(k_ideal) + " como melhor equilíbrio coesão-separação."}
""")

# =================================================================
# 7️⃣  K-MEANS FINAL
# =================================================================
kmeans_final = KMeans(n_clusters=k_ideal, init='k-means++',
                      random_state=12, n_init=10)
df['Cluster']  = kmeans_final.fit_predict(X)
CORES_CLUSTER  = {i: CORES_CLUSTER_PALETTE[i] for i in range(k_ideal)}
GUERRA_LABEL   = {0: "Sem Guerra", 1: "Com Guerra"}

# =================================================================
# 8️⃣  GRÁFICO 1 — Painel dos 3 índices
# =================================================================
header("SALVANDO GRÁFICOS")

fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle(
    f"Índices de Seleção do K  —  K={k_ideal} selecionado ({max_votos}/3 votos)",
    fontsize=13, color="#E6EDF3", y=1.02)

cfg_idx = [
    ("inertia",        "Elbow (Inércia WCSS)",  "#58A6FF", k_elbow),
    ("silhouette",     "Silhouette Score  ↑",   "#2ECC71", k_sil),
    ("davies_bouldin", "Davies-Bouldin  ↓",     "#E74C3C", k_db),
]
for ax, (col, title, color, k_best) in zip(axes1, cfg_idx):
    y = df_res[col].values
    ax.plot(k_range, y, 'o--', color=color, lw=2, ms=6,
            markerfacecolor="#F0A500", zorder=3)
    ax.fill_between(k_range, y, alpha=0.07, color=color)
    ax.scatter([k_best], [y[k_range.index(k_best)]], color="#F0A500",
               s=130, zorder=5, label=f"Melhor = K={k_best}")
    if k_best != k_ideal:
        ax.axvline(x=k_ideal, color="#FFFFFF", lw=1.2, linestyle=":",
                   alpha=0.6, label=f"K final = {k_ideal}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("K"); ax.set_xticks(k_range)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()

# =================================================================
# 9️⃣  GRÁFICO 2 — Análise FOB × Cluster × Guerra
# =================================================================
fob_perfil = df.groupby(['Cluster', 'Periodo_Guerra'])['Valor US$ FOB'].agg(
    Total='sum', Media='mean', Mediana='median', Std='std', Qtd='count'
).reset_index()
fob_perfil['Guerra_Label'] = fob_perfil['Periodo_Guerra'].map(GUERRA_LABEL)

fob_cluster     = df.groupby('Cluster')['Valor US$ FOB'].agg(Total_Cluster='sum').reset_index()
fob_total_geral = df['Valor US$ FOB'].sum()
fob_media_geral = df['Valor US$ FOB'].mean()

variacao = {}
for cluster in sorted(df['Cluster'].unique()):
    sub = fob_perfil[fob_perfil['Cluster'] == cluster].set_index('Periodo_Guerra')
    if 0 in sub.index and 1 in sub.index:
        v0 = sub.loc[0, 'Media']
        v1 = sub.loc[1, 'Media']
        variacao[cluster] = ((v1 - v0) / v0) * 100 if v0 != 0 else np.nan
    else:
        variacao[cluster] = np.nan



# Gráfico FOB — painel 4 subgráficos
fig_fob = plt.figure(figsize=(20, 16))
fig_fob.suptitle(f"Análise do Valor FOB por Cluster e Período de Guerra  (K={k_ideal})",
                 fontsize=15, color="#E6EDF3", y=0.99)
gs_fob = gridspec.GridSpec(2, 2, figure=fig_fob, hspace=0.5, wspace=0.38)

x     = np.arange(k_ideal)
width = 0.35

# A) FOB Médio
ax_a = fig_fob.add_subplot(gs_fob[0, 0])
pivot_media = fob_perfil.pivot_table(index='Cluster', columns='Guerra_Label', values='Media')
pivot_media = pivot_media.reindex(columns=["Sem Guerra", "Com Guerra"])
bars0 = ax_a.bar(x - width/2, pivot_media.get("Sem Guerra", 0),
                 width, color="#444C56", label="Sem Guerra", alpha=0.9)
bars1 = ax_a.bar(x + width/2, pivot_media.get("Com Guerra", 0),
                 width, color="#F0A500", label="Com Guerra", alpha=0.9)
ax_a.set_title("FOB Médio por Cluster e Período", fontsize=11)
ax_a.set_xlabel("Cluster"); ax_a.set_ylabel("FOB Médio (US$)")
ax_a.set_xticks(x); ax_a.set_xticklabels([f"C{i}" for i in range(k_ideal)])
ax_a.legend(); ax_a.grid(True, axis='y', alpha=0.25)
ax_a.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6
                      else f"${v/1e3:.0f}K" if v >= 1e3 else f"${v:.0f}"))
for bar in list(bars0) + list(bars1):
    h = bar.get_height()
    if h > 0:
        ax_a.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                  fmt_fob(h), ha='center', va='bottom', fontsize=7.5, color="#8B949E")

# B) FOB Total
ax_b = fig_fob.add_subplot(gs_fob[0, 1])
pivot_total = fob_perfil.pivot_table(index='Cluster', columns='Guerra_Label', values='Total')
pivot_total = pivot_total.reindex(columns=["Sem Guerra", "Com Guerra"])
bars0t = ax_b.bar(x - width/2, pivot_total.get("Sem Guerra", 0),
                  width, color="#444C56", label="Sem Guerra", alpha=0.9)
bars1t = ax_b.bar(x + width/2, pivot_total.get("Com Guerra", 0),
                  width, color="#F0A500", label="Com Guerra", alpha=0.9)
ax_b.set_title("FOB Total por Cluster e Período", fontsize=11)
ax_b.set_xlabel("Cluster"); ax_b.set_ylabel("FOB Total (US$)")
ax_b.set_xticks(x); ax_b.set_xticklabels([f"C{i}" for i in range(k_ideal)])
ax_b.legend(); ax_b.grid(True, axis='y', alpha=0.25)
ax_b.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"${v/1e9:.1f}Bi" if v >= 1e9
                      else f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}K"))
for bar in list(bars0t) + list(bars1t):
    h = bar.get_height()
    if h > 0:
        ax_b.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                  fmt_fob(h), ha='center', va='bottom', fontsize=7.5, color="#8B949E")

# C) Boxplot
ax_c = fig_fob.add_subplot(gs_fob[1, 0])
plot_data, labels_box, colors_box = [], [], []
for cluster in sorted(df['Cluster'].unique()):
    for guerra in [0, 1]:
        vals = df[(df['Cluster'] == cluster) &
                  (df['Periodo_Guerra'] == guerra)]['Valor US$ FOB'].dropna()
        if len(vals):
            plot_data.append(vals)
            labels_box.append(f"C{cluster}\n{'Gue' if guerra else 'Paz'}")
            colors_box.append("#F0A500" if guerra == 1 else "#444C56")
bp = ax_c.boxplot(plot_data, patch_artist=True, showfliers=False,
                  medianprops=dict(color="#E6EDF3", linewidth=2))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax_c.set_title("Distribuição FOB  (sem outliers extremos)", fontsize=11)
ax_c.set_xticklabels(labels_box, fontsize=8)
ax_c.set_ylabel("Valor US$ FOB")
ax_c.grid(True, axis='y', alpha=0.25)
ax_c.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6
                      else f"${v/1e3:.0f}K" if v >= 1e3 else f"${v:.0f}"))
ax_c.legend(handles=[Patch(color="#444C56", alpha=0.75, label="Sem Guerra"),
                      Patch(color="#F0A500", alpha=0.75, label="Com Guerra")], fontsize=9)

# D) Variação %
ax_d = fig_fob.add_subplot(gs_fob[1, 1])
clusters_var = sorted(variacao.keys())
vars_val     = [variacao[c] for c in clusters_var]
bar_colors   = ["#2ECC71" if v >= 0 else "#E74C3C" for v in vars_val]
bars_var     = ax_d.bar([f"Cluster {c}" for c in clusters_var],
                        vars_val, color=bar_colors, alpha=0.85, edgecolor="none")
ax_d.axhline(0, color="#8B949E", linewidth=1, linestyle="--")
ax_d.set_title("Δ% FOB Médio  (Com Guerra vs. Sem Guerra)", fontsize=11)
ax_d.set_ylabel("Variação (%)")
ax_d.grid(True, axis='y', alpha=0.25)
for bar, val in zip(bars_var, vars_val):
    ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 1.5
    ax_d.text(bar.get_x() + bar.get_width()/2, ypos,
              f"{'+' if val>=0 else ''}{val:.1f}%",
              ha='center', va='bottom', fontsize=10,
              color="#2ECC71" if val >= 0 else "#E74C3C", fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# =================================================================
# 🔟  GRÁFICO 3 — Estados × Cluster × Guerra
# =================================================================
estado_guerra  = (
    df.groupby(['Cluster', 'UF do Produto', 'Periodo_Guerra'])
    .size().reset_index(name='Qtd')
)
totais_cluster = df.groupby('Cluster').size()

header(f"ESTADOS POR CLUSTER × PERÍODO DE GUERRA  (K={k_ideal})")

for cluster in sorted(df['Cluster'].unique()):
    cor     = ["32", "31", "34", "33", "35"][cluster % 5]
    total_c = totais_cluster[cluster]
    subheader(f"Cluster {cluster}  ({total_c:,} registros)", char="━")
    sub     = estado_guerra[estado_guerra['Cluster'] == cluster]

    print(f"\n  {'UF':>4}  {'Guerra':^8}  {'Qtd':>7}  {'% no cluster':<28}  {'% do UF':>7}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*28}  {'─'*7}")

    for uf in sorted(sub['UF do Produto'].unique()):
        sub_uf   = sub[sub['UF do Produto'] == uf]
        total_uf = sub_uf['Qtd'].sum()
        for _, row in sub_uf.sort_values('Periodo_Guerra').iterrows():
            g_lbl = "\033[91mSim\033[0m" if row['Periodo_Guerra'] == 1 else "\033[92mNão\033[0m"
            barra = bar_ascii(row['Qtd'] / total_c, width=16)
            print(f"  \033[{cor}m{uf:>4}\033[0m  {g_lbl:^16}  "
                  f"{int(row['Qtd']):>7,}  {barra}  {row['Qtd']/total_uf:>6.1%}")

fig3, axes3 = plt.subplots(1, k_ideal, figsize=(7*k_ideal, 8))
if k_ideal == 1: axes3 = [axes3]
fig3.suptitle(f"Registros por Estado, Cluster e Período de Guerra  (K={k_ideal})",
              fontsize=14, color="#E6EDF3", y=1.01)

for i, cluster in enumerate(sorted(df['Cluster'].unique())):
    ax  = axes3[i]
    sub = estado_guerra[estado_guerra['Cluster'] == cluster].copy()
    sub['Guerra_Label'] = sub['Periodo_Guerra'].map({0: "Sem Guerra", 1: "Com Guerra"})
    pivot = sub.pivot_table(index='UF do Produto', columns='Guerra_Label',
                            values='Qtd', aggfunc='sum', fill_value=0)
    pivot = pivot.reindex(columns=["Sem Guerra", "Com Guerra"], fill_value=0)
    pivot = (pivot.assign(T=pivot.sum(axis=1))
             .sort_values("T", ascending=True).drop(columns="T"))
    pivot.plot(kind='barh', ax=ax,
               color=["#444C56", CORES_CLUSTER[cluster]],
               width=0.65, edgecolor="none", stacked=True)
    ax.set_title(f"Cluster {cluster}", fontsize=13, pad=10)
    ax.set_xlabel("Registros"); ax.set_ylabel("")
    ax.grid(True, axis='x', alpha=0.2)
    ax.legend(title="Período", fontsize=9)
    totals = pivot.sum(axis=1)
    for y_pos, total in enumerate(totals):
        ax.text(total + totals.max()*0.01, y_pos,
                f"{int(total):,}", va='center', fontsize=7.5, color="#8B949E")

plt.tight_layout()
plt.show()

# =================================================================
# 1️⃣1️⃣  GRÁFICO 4 — PCA
# =================================================================
pca    = PCA(n_components=2)
X_pca  = pca.fit_transform(X)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster']        = df['Cluster']
df_pca['Periodo_Guerra'] = df['Periodo_Guerra']

fig4, ax4 = plt.subplots(figsize=(13, 8))
for cid in range(k_ideal):
    for guerra, marker, alpha in [(0, 'o', 0.45), (1, 'X', 0.80)]:
        mask = (df_pca['Cluster'] == cid) & (df_pca['Periodo_Guerra'] == guerra)
        sub  = df_pca[mask]
        ax4.scatter(sub['PC1'], sub['PC2'],
                    c=CORES_CLUSTER[cid], marker=marker,
                    alpha=alpha, s=55, edgecolors='none',
                    label=f"Cluster {cid} | {'Com' if guerra else 'Sem'} Guerra")

ax4.set_title(f"Mapa PCA — K={k_ideal}", fontsize=14, pad=14)
ax4.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]:.2%} var.)")
ax4.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]:.2%} var.)")
ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
