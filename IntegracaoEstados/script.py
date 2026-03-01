##
## Análise de Clusterização: Integração Comercial dos Estados Brasileiros
## Objetivo: Verificar se estados do Norte/Nordeste são mais integrados a economias emergentes, enquanto Sul/Sudeste aos desenvolvidos.
## Dados: Exportações brasileiras por UF, país destino, bloco econômico (2025) - ComexStat.


import warnings
warnings.filterwarnings('ignore')

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Evita erro do wmic no Windows 11

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from sklearn.metrics.pairwise import cosine_similarity

def cluster_purity(cluster_labels, true_labels):
    """Pureza: fração de pontos que pertencem à classe majoritária em cada cluster."""
    contingency = pd.crosstab(cluster_labels, true_labels)
    return contingency.max(axis=1).sum() / len(cluster_labels)


def cluster_entropy(cluster_labels, true_labels):
    """Entropia média ponderada dos clusters em relação a rótulos verdadeiros."""
    n = len(cluster_labels)
    weighted_entropy = 0.0
    for k in np.unique(cluster_labels):
        mask = cluster_labels == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        probs = pd.Series(true_labels[mask]).value_counts(normalize=True)
        probs = probs[probs > 0]
        ent_k = -(probs * np.log2(probs)).sum()
        weighted_entropy += (n_k / n) * ent_k
    return weighted_entropy


def draw_confidence_ellipse(ax, x, y, color, n_std=2.0):
    """Desenha elipse de confiança ao redor de um cluster (baseada em covariância)."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_rx = np.sqrt(1 + pearson)
    ell_ry = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_rx * 2, height=ell_ry * 2,
        facecolor=color, alpha=0.13,
        edgecolor=color, linewidth=2.0, linestyle='--',
        zorder=1
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(np.mean(x), np.mean(y)))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

# Configurações de visualização
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight'
})
sns.set_style("whitegrid")

OUTPUT_DIR = "resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. COLETA E CARREGAMENTO DOS DADOS
print("1. COLETA E CARREGAMENTO DOS DADOS")

df1 = pd.read_csv('exp_america_norte_europa.csv', sep=';', encoding='utf-8')
df2 = pd.read_csv('exp_america_sul.csv', sep=';', encoding='utf-8')
df3 = pd.read_csv('exp_asia.csv', sep=';', encoding='utf-8')
df4 = pd.read_csv('exp_africa_e_oceania.csv', sep=';', encoding='utf-8')

print(f"  América do Norte/Europa: {df1.shape[0]:>10,} registros")
print(f"  América do Sul:          {df2.shape[0]:>10,} registros")
print(f"  Ásia:                    {df3.shape[0]:>10,} registros")
print(f"  África e Oceania:        {df4.shape[0]:>10,} registros")

df = pd.concat([df1, df2, df3, df4], ignore_index=True)
print(f"  Total unificado:         {df.shape[0]:>10,} registros")

# 2. PRÉ-PROCESSAMENTO
print("2. PRÉ-PROCESSAMENTO")

# 2.1 Limpeza de strings (remover \r, espaços extras)
str_cols = ['UF do Produto', 'Países', 'Via', 'Descrição NCM',
            'Bloco Econômico']
for col in str_cols:
    df[col] = df[col].astype(str).str.replace('\r', '', regex=False).str.strip()

df['Valor US$ FOB'] = pd.to_numeric(df['Valor US$ FOB'], errors='coerce')

# 2.2 Verificar valores faltantes
print("\n  Valores faltantes por coluna:")
missing = df.isnull().sum()
for col, n in missing.items():
    pct = n / len(df) * 100
    status = f"  {n:,} ({pct:.2f}%)" if n > 0 else "   OK"
    print(f"    {col:<25s} {status}")

# Remover "Não Declarada" e valores nulos
n_before = len(df)
df = df[df['UF do Produto'] != 'Não Declarada']
df = df.dropna(subset=['Valor US$ FOB', 'UF do Produto'])
n_after = len(df)
print(f"\n  Registros removidos (Não Declarada/NaN): {n_before - n_after:,}")
print(f"  Registros restantes: {n_after:,}")

# 2.3 Classificação de regiões brasileiras
REGIOES = {
    'Norte': ['Acre', 'Amapá', 'Amazonas', 'Pará', 'Rondônia', 'Roraima', 'Tocantins'],
    'Nordeste': ['Alagoas', 'Bahia', 'Ceará', 'Maranhão', 'Paraíba',
                 'Pernambuco', 'Piauí', 'Rio Grande do Norte', 'Sergipe'],
    'Centro-Oeste': ['Distrito Federal', 'Goiás', 'Mato Grosso', 'Mato Grosso do Sul'],
    'Sudeste': ['Espírito Santo', 'Minas Gerais', 'Rio de Janeiro', 'São Paulo'],
    'Sul': ['Paraná', 'Rio Grande do Sul', 'Santa Catarina']
}

uf_to_regiao = {}
for regiao, ufs in REGIOES.items():
    for uf in ufs:
        uf_to_regiao[uf] = regiao

df['Região'] = df['UF do Produto'].map(uf_to_regiao)

# 2.4 Classificação de economias: Desenvolvidas vs Emergentes
PAISES_DESENVOLVIDOS = [
    'Estados Unidos', 'Canadá', 'Alemanha', 'França', 'Reino Unido',
    'Itália', 'Espanha', 'Países Baixos (Holanda)', 'Bélgica', 'Suíça',
    'Suécia', 'Noruega', 'Dinamarca', 'Finlândia', 'Áustria', 'Irlanda',
    'Portugal', 'Grécia', 'Luxemburgo', 'Islândia', 'Japão',
    'Coreia do Sul', 'Austrália', 'Nova Zelândia', 'Singapura',
    'Israel', 'República Tcheca', 'Polônia', 'Hungria', 'Eslovênia',
    'Estônia', 'Letônia', 'Lituânia', 'Eslováquia', 'Croácia',
    'Malta', 'Chipre', 'Taiwan (Formosa)'
]

df['Tipo Economia'] = df['Países'].apply(
    lambda x: 'Desenvolvida' if x in PAISES_DESENVOLVIDOS else 'Emergente'
)

print(f"\n  Destinos classificados como Desenvolvidos: "
      f"{df[df['Tipo Economia']=='Desenvolvida']['Países'].nunique()}")
print(f"  Destinos classificados como Emergentes:     "
      f"{df[df['Tipo Economia']=='Emergente']['Países'].nunique()}")

# 2.5 Classificação por Bloco Econômico (mais granular)
blocos_unicos = df['Bloco Econômico'].unique()
print(f"\n  Blocos econômicos encontrados: {blocos_unicos}")

# 2.6 Produtos mais exportados por Macrorregião
print("\n  --- Produtos mais exportados por Macrorregião ---")
_macro_map = {
    'Norte': 'Norte/Nordeste', 'Nordeste': 'Norte/Nordeste',
    'Centro-Oeste': 'Centro-Oeste',
    'Sudeste': 'Sul/Sudeste', 'Sul': 'Sul/Sudeste'
}
df['_Macro'] = df['Região'].map(_macro_map)

for macro in ['Sul/Sudeste', 'Norte/Nordeste']:
    subset = df[df['_Macro'] == macro]
    top_prod = (
        subset.groupby('Descrição NCM')['Valor US$ FOB']
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    total_macro = subset['Valor US$ FOB'].sum()
    print(f"\n  Top 5 produtos exportados — {macro}:")
    print(f"  {'#':<3} {'Produto':<65} {'US$ FOB':>18} {'%Total':>8}")
    print(f"  {'-'*3} {'-'*65} {'-'*18} {'-'*8}")
    for i, (prod, val) in enumerate(top_prod.items(), 1):
        prod_short = (prod[:62] + '...') if len(prod) > 65 else prod
        pct = val / total_macro * 100
        print(f"  {i:<3} {prod_short:<65} US$ {val:>13,.0f} {pct:>7.2f}%")

df.drop(columns=['_Macro'], inplace=True)

# 3. CONSTRUÇÃO DA MATRIZ DE FEATURES POR ESTADO
print("3. CONSTRUÇÃO DA MATRIZ DE FEATURES")

# Feature 1: % do valor exportado por Bloco Econômico
pivot_bloco = df.pivot_table(
    index='UF do Produto',
    columns='Bloco Econômico',
    values='Valor US$ FOB',
    aggfunc='sum',
    fill_value=0
)

# Normalizar para proporções (cada linha soma 1)
pivot_bloco_pct = pivot_bloco.div(pivot_bloco.sum(axis=1), axis=0) * 100
pivot_bloco_pct.columns = [f'%_{col}' for col in pivot_bloco_pct.columns]

# Feature 2: % exportado para economias Desenvolvidas vs Emergentes
pivot_tipo = df.pivot_table(
    index='UF do Produto',
    columns='Tipo Economia',
    values='Valor US$ FOB',
    aggfunc='sum',
    fill_value=0
)
pivot_tipo_pct = pivot_tipo.div(pivot_tipo.sum(axis=1), axis=0) * 100
pivot_tipo_pct.columns = [f'%_{col}' for col in pivot_tipo_pct.columns]

# Feature 3: % para top países (China, EUA, Argentina)
top_paises = ['China', 'Estados Unidos', 'Argentina', 'Índia',
              'Países Baixos (Holanda)', 'Chile', 'Japão', 'México']
for pais in top_paises:
    total_uf = df.groupby('UF do Produto')['Valor US$ FOB'].sum()
    pais_uf = df[df['Países'] == pais].groupby('UF do Produto')['Valor US$ FOB'].sum()
    col_name = f'%_{pais}'
    pivot_bloco_pct[col_name] = (pais_uf / total_uf * 100).fillna(0)

# Feature 4: Diversificação (nº de países destino)
n_paises = df.groupby('UF do Produto')['Países'].nunique()
pivot_bloco_pct['N_Países_Destino'] = n_paises

# Feature 5: Valor total exportado (log-transformado)
total_exp = df.groupby('UF do Produto')['Valor US$ FOB'].sum()
pivot_bloco_pct['Log_Valor_Total'] = np.log1p(total_exp)

# Feature 6: HHI (concentração de destinos)
def calc_hhi(group):
    shares = group / group.sum()
    return (shares ** 2).sum()

hhi = df.groupby('UF do Produto').apply(
    lambda x: calc_hhi(x.groupby('Países')['Valor US$ FOB'].sum())
)
pivot_bloco_pct['HHI_Destinos'] = hhi

# Merge com tipo de economia
features = pivot_bloco_pct.join(pivot_tipo_pct)

# Adicionar metadados (região)
features['Região'] = features.index.map(uf_to_regiao)
features['Macro_Região'] = features['Região'].map({
    'Norte': 'Norte/Nordeste',
    'Nordeste': 'Norte/Nordeste',
    'Centro-Oeste': 'Centro-Oeste',
    'Sudeste': 'Sul/Sudeste',
    'Sul': 'Sul/Sudeste'
})

print(f"\n  Matriz de features: {features.shape[0]} estados × "
      f"{features.select_dtypes(include=[np.number]).shape[1]} features numéricas")
print(f"\n  Features utilizadas:")
for i, col in enumerate(features.select_dtypes(include=[np.number]).columns, 1):
    print(f"    {i:2d}. {col}")

# DataFrame numérico para análises
X_df = features.select_dtypes(include=[np.number]).copy()

# Verificar e remover features com variância zero
low_var = X_df.columns[X_df.var() < 1e-10]
if len(low_var) > 0:
    print(f"\n   Features com variância ~0 removidas: {list(low_var)}")
    X_df = X_df.drop(columns=low_var)

print(f"\n  Amostra da matriz (primeiros 5 estados):")
print(X_df.head().round(2).to_string())

# 4. ANÁLISE EXPLORATÓRIA (EDA)
print("4. ANÁLISE EXPLORATÓRIA (EDA)")

# 4.1 Estatísticas descritivas
desc = X_df.describe().T
desc['cv'] = desc['std'] / desc['mean']  # Coeficiente de variação
print("\n  Estatísticas descritivas:")
print(desc[['mean', 'std', 'min', '50%', 'max', 'cv']].round(3).to_string())

# Salvar
desc.to_csv(f'{OUTPUT_DIR}/estatisticas_descritivas.csv', sep=';')

# 4.3 Boxplots por macrorregião
key_features = ['%_Desenvolvida', '%_Emergente', '%_China', '%_Estados Unidos']
key_features = [f for f in key_features if f in X_df.columns]

fig, axes = plt.subplots(1, len(key_features), figsize=(5 * len(key_features), 6))
if len(key_features) == 1:
    axes = [axes]
for i, feat in enumerate(key_features):
    data_plot = features[[feat, 'Macro_Região']].dropna()
    order = ['Norte/Nordeste', 'Centro-Oeste', 'Sul/Sudeste']
    sns.boxplot(data=data_plot, x='Macro_Região', y=feat, ax=axes[i],
                order=order, palette='Set2')
    axes[i].set_title(feat, fontsize=12)
    axes[i].tick_params(axis='x', rotation=30)
plt.suptitle('Distribuição por Macrorregião', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/boxplots_macroregioes.png')
plt.close()
print("   Boxplots salvos")

# 4.4 Matriz de correlação (Pearson e Spearman)
for method in ['pearson', 'spearman']:
    corr = X_df.corr(method=method)
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={'size': 7}, linewidths=0.5)
    ax.set_title(f'Matriz de Correlação ({method.title()})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/correlacao_{method}.png')
    plt.close()
print("   Matrizes de correlação salvas (Pearson e Spearman)")

# 4.5 Identificar variáveis altamente correlacionadas
corr_pearson = X_df.corr(method='pearson')
high_corr_pairs = []
for i in range(len(corr_pearson.columns)):
    for j in range(i + 1, len(corr_pearson.columns)):
        if abs(corr_pearson.iloc[i, j]) > 0.85:
            high_corr_pairs.append(
                (corr_pearson.columns[i], corr_pearson.columns[j],
                 corr_pearson.iloc[i, j])
            )
if high_corr_pairs:
    print("\n   Pares altamente correlacionados (|r| > 0.85):")
    for c1, c2, r in high_corr_pairs:
        print(f"    {c1} × {c2}: r = {r:.3f}")
else:
    print("\n   Nenhum par com correlação > 0.85")

# 4.6 Detecção de outliers (IQR)
print("\n  Outliers por feature (método IQR):")
outlier_counts = {}
for col in X_df.columns:
    Q1 = X_df[col].quantile(0.25)
    Q3 = X_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = X_df[(X_df[col] < lower) | (X_df[col] > upper)]
    if len(outliers) > 0:
        outlier_counts[col] = len(outliers)
        print(f"    {col}: {len(outliers)} outlier(s) — "
              f"{list(outliers.index[:3])}")

# 5. REDUÇÃO DE DIMENSIONALIDADE
print("5. REDUÇÃO DE DIMENSIONALIDADE")

# Padronização (Z-Score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
X_scaled_df = pd.DataFrame(X_scaled, index=X_df.index, columns=X_df.columns)

# 5.1 PCA — Variância Explicada
pca_full = PCA()
pca_full.fit(X_scaled)

var_explained = pca_full.explained_variance_ratio_
var_cumulative = np.cumsum(var_explained)

print("\n  PCA — Variância Explicada:")
for i, (ve, vc) in enumerate(zip(var_explained, var_cumulative)):
    print(f"    PC{i+1}: {ve:.4f} ({vc:.4f} acumulada)")
    if vc > 0.95:
        n_components_95 = i + 1
        print(f"{n_components_95} componentes explicam >=95% da variância")
        break

# Gráfico de variância explicada
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(range(1, len(var_explained) + 1), var_explained, alpha=0.7,
        color='steelblue', label='Individual')
ax1.plot(range(1, len(var_cumulative) + 1), var_cumulative, 'ro-',
         label='Acumulada')
ax1.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, label='95%')
ax1.set_xlabel('Componente Principal')
ax1.set_ylabel('Variância Explicada')
ax1.set_title('PCA — Scree Plot')
ax1.legend()

# Biplot (PC1 vs PC2)
pca_2d = PCA(n_components=2)
X_pca2 = pca_2d.fit_transform(X_scaled)

colors_map = {'Norte/Nordeste': '#e74c3c', 'Centro-Oeste': '#f39c12', 'Sul/Sudeste': '#2ecc71'}
for macro, color in colors_map.items():
    mask = features['Macro_Região'] == macro
    ax2.scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=color, label=macro,
                s=80, edgecolors='black', alpha=0.8, zorder=3)
for i, uf in enumerate(X_df.index):
    ax2.annotate(uf, (X_pca2[i, 0], X_pca2[i, 1]),
                 fontsize=6, ha='center', va='bottom', alpha=0.7)

ax2.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
ax2.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
ax2.set_title('PCA — Projeção 2D por Macrorregião')
ax2.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pca_scree_biplot.png')
plt.close()
print("   PCA scree plot e biplot salvos")

# Loadings do PCA
loadings = pd.DataFrame(
    pca_full.components_[:6].T,
    columns=['PC1', 'PC2', 'PC3','PC4','PC5','PC6'],
    index=X_df.columns
)
print("\n  Loadings PCA (primeiras 6 componentes):")
print(loadings.round(3).to_string())
loadings.to_csv(f'{OUTPUT_DIR}/pca_loadings.csv', sep=';')

# 5.2 t-SNE
print("\n  Executando t-SNE...")
n_samples = X_scaled.shape[0]
perp = min(15, max(5, n_samples // 2))  # Perplexity maior = melhor estrutura global
tsne = TSNE(n_components=2, random_state=42, perplexity=perp,
            max_iter=3000, learning_rate=200, init='pca', n_jobs=1)
X_tsne = tsne.fit_transform(X_scaled)

print("   t-SNE calculado (gráfico será gerado após clustering)")

# Preparar espaço para clustering via PCA (reduz ruído e maldizão da dimensionalidade)
_n_pca_clust = max(2, int(np.where(var_cumulative >= 0.80)[0][0]) + 1)
_pca_clust_model = PCA(n_components=_n_pca_clust, random_state=42)
X_for_clust = _pca_clust_model.fit_transform(X_scaled)

# 6. CLUSTERIZAÇÃO
print("6. CLUSTERIZAÇÃO")
print(f"\n   Clustering em {_n_pca_clust} componentes PCA "
      f"({var_cumulative[_n_pca_clust-1]:.1%} da variância explicada)")

# 6.1 Determinação do K ótimo — Elbow (SSE) + Silhouette + Pureza + Entropia
# Rótulos verdadeiros (Macrorregião) usados como referência para pureza/entropia
true_labels_macro = features['Macro_Região'].values

K_range = range(2, min(10, n_samples))
metrics = {
    'K': [], 'Inertia': [], 'Silhouette': [],
    'Pureza': [], 'Entropia': []
}

for k in K_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
    labels = km.fit_predict(X_for_clust)
    metrics['K'].append(k)
    metrics['Inertia'].append(km.inertia_)
    metrics['Silhouette'].append(silhouette_score(X_for_clust, labels))
    metrics['Pureza'].append(cluster_purity(labels, true_labels_macro))
    metrics['Entropia'].append(cluster_entropy(labels, true_labels_macro))

metrics_df = pd.DataFrame(metrics)
print("\n  Métricas por K:")
print(metrics_df.round(4).to_string(index=False))
metrics_df.to_csv(f'{OUTPUT_DIR}/metricas_k.csv', sep=';', index=False)

# Gráfico das métricas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(metrics['K'], metrics['Inertia'], 'bo-', linewidth=2)
axes[0, 0].set_title('Método Elbow — SSE (Inércia)')
axes[0, 0].set_xlabel('K')
axes[0, 0].set_ylabel('SSE (Inércia)')

axes[0, 1].plot(metrics['K'], metrics['Silhouette'], 'go-', linewidth=2)
axes[0, 1].axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='Referência (0.25)')
axes[0, 1].set_title('Silhouette Score (maior = melhor)')
axes[0, 1].set_xlabel('K')
axes[0, 1].set_ylabel('Silhouette')
axes[0, 1].legend()

axes[1, 0].plot(metrics['K'], metrics['Pureza'], 'ro-', linewidth=2)
axes[1, 0].set_title('Pureza dos Clusters (maior = melhor)')
axes[1, 0].set_xlabel('K')
axes[1, 0].set_ylabel('Pureza')

axes[1, 1].plot(metrics['K'], metrics['Entropia'], 'mo-', linewidth=2)
axes[1, 1].set_title('Entropia dos Clusters (menor = melhor)')
axes[1, 1].set_xlabel('K')
axes[1, 1].set_ylabel('Entropia (bits)')

plt.suptitle('Avaliação do Número Ótimo de Clusters', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/metricas_clusters.png')
plt.close()
print("\n   Gráficos de métricas salvos")

# Selecionar K ótimo (melhor silhouette)
best_k = metrics_df.loc[metrics_df['Silhouette'].idxmax(), 'K']
best_k = int(best_k)
print(f"\n   K ótimo selecionado (max Silhouette): K = {best_k}")

# 6.2 K-Means com K ótimo
km_best = KMeans(n_clusters=best_k, n_init=30, random_state=42, max_iter=500)
labels_best = km_best.fit_predict(X_for_clust)
features['Cluster_KMeans'] = labels_best

pur = cluster_purity(labels_best, true_labels_macro)
ent = cluster_entropy(labels_best, true_labels_macro)
print(f"\n  K-Means (K={best_k}):")
print(f"    SSE (Inércia):       {km_best.inertia_:.4f}")
print(f"    Silhouette Score:    {silhouette_score(X_for_clust, labels_best):.4f}")
print(f"    Pureza:              {pur:.4f}")
print(f"    Entropia:            {ent:.4f} bits")

# --- Matriz de Similaridade (Cosseno) ---
print("\n  --- Matriz de Similaridade ---")
order_idx = np.argsort(labels_best)
X_sim_ordered = X_scaled[order_idx]
labels_sim_ordered = labels_best[order_idx]
uf_sim_ordered = X_df.index[order_idx]

sim_matrix = cosine_similarity(X_sim_ordered)
sim_df = pd.DataFrame(sim_matrix, index=uf_sim_ordered, columns=uf_sim_ordered)

fig, ax = plt.subplots(figsize=(14, 11))
cluster_sizes_sim = [int(np.sum(labels_sim_ordered == k)) for k in range(best_k)]
sns.heatmap(sim_df, cmap='YlOrRd', ax=ax, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 6},
            xticklabels=True, yticklabels=True, linewidths=0.3)
cumulative_sim = 0
for size in cluster_sizes_sim[:-1]:
    cumulative_sim += size
    ax.axhline(cumulative_sim, color='blue', linewidth=2.5)
    ax.axvline(cumulative_sim, color='blue', linewidth=2.5)
ax.set_title(f'Matriz de Similaridade (Cosseno) — Ordenada por Cluster (K={best_k})', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/matriz_similaridade.png')
plt.close()
print("   Matriz de similaridade salva")

# Paleta de cores para clusters (suporta até K=9)
CLUSTER_COLORS = [
    '#E74C3C', '#1A6FBF', '#2ECC71', '#F39C12', '#9B59B6',
    '#1ABC9C', '#E67E22', '#2C3E50', '#F06292'
]

# Visualizações t-SNE coloridas por Cluster
cluster_palette = CLUSTER_COLORS[:best_k]

fig, ax = plt.subplots(figsize=(10, 8))
for k in range(best_k):
    mask = labels_best == k
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=[cluster_palette[k]], label=f'Cluster {k}',
               s=120, edgecolors='black', alpha=0.85, zorder=3)
    draw_confidence_ellipse(ax, X_tsne[mask, 0], X_tsne[mask, 1], cluster_palette[k])
for i, uf in enumerate(X_df.index):
    ax.annotate(uf, (X_tsne[i, 0], X_tsne[i, 1]),
                fontsize=7, ha='center', va='bottom')
ax.set_title('t-SNE — Visualização 2D dos Estados (por Cluster)')
ax.legend(title='Cluster')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tsne_2d.png')
plt.close()
print("   t-SNE (por cluster) salvo")

# 6.4 Estabilidade do K-Means
print("\n  --- Teste de Estabilidade ---")
seeds = [0, 1, 2, 42, 100]
all_labels = []
for seed in seeds:
    km_s = KMeans(n_clusters=best_k, n_init=20, random_state=seed, max_iter=500)
    all_labels.append(km_s.fit_predict(X_for_clust))

# ARI e NMI entre todos os pares de execuções
ari_scores = []
nmi_scores = []
for i in range(len(seeds)):
    for j in range(i + 1, len(seeds)):
        ari = adjusted_rand_score(all_labels[i], all_labels[j])
        nmi = normalized_mutual_info_score(all_labels[i], all_labels[j])
        ari_scores.append(ari)
        nmi_scores.append(nmi)
        print(f"    Seeds {seeds[i]} vs {seeds[j]}: "
              f"ARI = {ari:.4f}, NMI = {nmi:.4f}")

mean_ari = np.mean(ari_scores)
mean_nmi = np.mean(nmi_scores)
print(f"\n    Média ARI: {mean_ari:.4f} "
      f"{' Estável (>0.8)' if mean_ari > 0.8 else ' Instável (<0.8)'}")
print(f"    Média NMI: {mean_nmi:.4f}")

# 6.5 Clustering Hierárquico (Aglomerativo)
print("\n  --- Clustering Hierárquico ---")
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X_for_clust, method='ward')
fig, ax = plt.subplots(figsize=(16, 8))
dendrogram(Z, labels=X_df.index.tolist(), leaf_rotation=90,
           leaf_font_size=9, ax=ax, color_threshold=0)
ax.set_title('Dendrograma — Agrupamento Hierárquico')
ax.set_ylabel('Distância')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/dendrograma.png')
plt.close()
print("   Dendrograma salvo")

agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
labels_agg = agg.fit_predict(X_for_clust)
features['Cluster_Hierarquico'] = labels_agg

sil_agg = silhouette_score(X_for_clust, labels_agg)
print(f"    Silhouette (Hierárquico, K={best_k}): {sil_agg:.4f}")

# 6.6 DBSCAN
print("\n  --- DBSCAN ---")
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=3)
nn.fit(X_for_clust)
distances, _ = nn.kneighbors(X_for_clust)
distances = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(distances, 'b-', linewidth=1.5)
ax.set_title('k-Distance Graph (para escolha de eps)')
ax.set_xlabel('Pontos ordenados')
ax.set_ylabel('3-NN Distance')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/kdistance.png')
plt.close()

# Testar vários eps
eps_candidates = np.percentile(distances, [50, 60, 70, 80])
best_dbscan_sil = -1
best_dbscan_labels = None
best_eps = None

for eps in eps_candidates:
    db = DBSCAN(eps=eps, min_samples=2)
    db_labels = db.fit_predict(X_for_clust)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = (db_labels == -1).sum()
    if n_clusters_db >= 2:
        sil_db = silhouette_score(X_for_clust, db_labels)
        print(f"    eps={eps:.3f}: {n_clusters_db} clusters, "
              f"{n_noise} noise, silhouette={sil_db:.4f}")
        if sil_db > best_dbscan_sil:
            best_dbscan_sil = sil_db
            best_dbscan_labels = db_labels
            best_eps = eps
    else:
        print(f"    eps={eps:.3f}: {n_clusters_db} clusters, "
              f"{n_noise} noise — insuficiente")

if best_dbscan_labels is not None:
    features['Cluster_DBSCAN'] = best_dbscan_labels
    print(f"     Melhor DBSCAN: eps={best_eps:.3f}, "
          f"silhouette={best_dbscan_sil:.4f}")

# 6.7 Visualização dos clusters no PCA 2D
cluster_palette = CLUSTER_COLORS[:best_k]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# K-Means
for k in range(best_k):
    mask = labels_best == k
    pts = X_pca2[mask]
    axes[0].scatter(pts[:, 0], pts[:, 1],
                    c=[cluster_palette[k]], label=f'Cluster {k}',
                    s=100, edgecolors='black', alpha=0.85)
    draw_confidence_ellipse(axes[0], pts[:, 0], pts[:, 1], cluster_palette[k])
for i, uf in enumerate(X_df.index):
    axes[0].annotate(uf, (X_pca2[i, 0], X_pca2[i, 1]),
                     fontsize=6, ha='center', va='bottom')
axes[0].set_title(f'K-Means (K={best_k})')
axes[0].set_xlabel(f'PC1 ({var_explained[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({var_explained[1]:.1%})')
axes[0].legend(title='Cluster')

# Hierárquico
for k in range(best_k):
    mask = labels_agg == k
    pts = X_pca2[mask]
    axes[1].scatter(pts[:, 0], pts[:, 1],
                    c=[cluster_palette[k]], label=f'Cluster {k}',
                    s=100, edgecolors='black', alpha=0.85)
    draw_confidence_ellipse(axes[1], pts[:, 0], pts[:, 1], cluster_palette[k])
for i, uf in enumerate(X_df.index):
    axes[1].annotate(uf, (X_pca2[i, 0], X_pca2[i, 1]),
                     fontsize=6, ha='center', va='bottom')
axes[1].set_title(f'Hierárquico (K={best_k})')
axes[1].set_xlabel(f'PC1 ({var_explained[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({var_explained[1]:.1%})')
axes[1].legend(title='Cluster')

plt.suptitle(f'Clusters no Espaço PCA 2D (K={best_k})', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/clusters_pca2d.png')
plt.close()

print("\n   Visualização de clusters salva")

# 6.8 Mapa direto nas 2 features mais discriminantes
feat_x = '%_Desenvolvida' if '%_Desenvolvida' in X_df.columns else X_df.columns[0]
feat_y = '%_China' if '%_China' in X_df.columns else X_df.columns[1]

fig, ax = plt.subplots(figsize=(10, 7))
for k in range(best_k):
    mask = features['Cluster_KMeans'] == k
    xs = features.loc[mask, feat_x].values
    ys = features.loc[mask, feat_y].values
    ax.scatter(xs, ys, c=[CLUSTER_COLORS[k]], label=f'Cluster {k}',
               s=120, edgecolors='black', alpha=0.85, zorder=3)
    draw_confidence_ellipse(ax, xs, ys, CLUSTER_COLORS[k])
for uf in features.index:
    ax.annotate(uf, (features.loc[uf, feat_x], features.loc[uf, feat_y]),
                fontsize=7, ha='center', va='bottom')
ax.set_xlabel(feat_x, fontsize=12)
ax.set_ylabel(feat_y, fontsize=12)
ax.set_title('Separação dos Clusters — Features mais Discriminantes', fontsize=13)
ax.legend(title='Cluster')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/clusters_features_diretas.png')
plt.close()
print("   Scatter features diretas salvo")

# 7.1 Perfil médio de cada cluster
print("\n  --- Perfil Médio por Cluster (K-Means) ---")
cluster_profiles = features.groupby('Cluster_KMeans')[
    X_df.columns.tolist()
].mean()
print(cluster_profiles.round(2).to_string())
cluster_profiles.to_csv(f'{OUTPUT_DIR}/perfil_clusters.csv', sep=';')

# 7.2 Composição regional de cada cluster
print("\n  --- Composição Regional dos Clusters ---")
cross = pd.crosstab(features['Cluster_KMeans'], features['Região'])
print(cross.to_string())

cross_macro = pd.crosstab(features['Cluster_KMeans'], features['Macro_Região'])
print("\n  Por Macrorregião:")
print(cross_macro.to_string())

# Gráfico de composição
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cross.plot(kind='bar', stacked=True, ax=axes[0], colormap='Set2')
axes[0].set_title('Composição Regional dos Clusters')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Nº de Estados')
axes[0].legend(title='Região', fontsize=8, loc='upper right')

cross_macro.plot(kind='bar', stacked=True, ax=axes[1],
                 color=['#e74c3c', '#f39c12', '#2ecc71'])
axes[1].set_title('Composição por Macrorregião')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Nº de Estados')
axes[1].legend(title='Macrorregião', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/composicao_clusters.png')
plt.close()
print("\n   Gráficos de composição salvos")

# 7.3 Tabela detalhada: cada estado + cluster + região
tabela_final = features[['Região', 'Macro_Região', 'Cluster_KMeans',
                         'Cluster_Hierarquico']].copy()
tabela_final['%_Desenvolvida'] = features['%_Desenvolvida'].round(1)
tabela_final['%_Emergente'] = features['%_Emergente'].round(1)
if '%_China' in features.columns:
    tabela_final['%_China'] = features['%_China'].round(1)
if '%_Estados Unidos' in features.columns:
    tabela_final['%_EUA'] = features['%_Estados Unidos'].round(1)

tabela_final = tabela_final.sort_values(['Cluster_KMeans', 'Região'])
print("\n  --- Tabela Final: Estado  Cluster ---")
print(tabela_final.to_string())
tabela_final.to_csv(f'{OUTPUT_DIR}/tabela_estados_clusters.csv', sep=';')

print(f"""
  • {features.shape[0]} estados analisados com {X_df.shape[1]} features
  • K ótimo: {best_k} clusters
      SSE (Inércia):    {km_best.inertia_:.4f}
      Silhouette:       {silhouette_score(X_for_clust, labels_best):.4f}
      Pureza:           {cluster_purity(labels_best, true_labels_macro):.4f}
      Entropia:         {cluster_entropy(labels_best, true_labels_macro):.4f} bits
  • Estabilidade: ARI médio = {mean_ari:.4f}
  • PCA: {n_components_95} componentes explicam >=95% da variância
  • Todos os gráficos e tabelas salvos em: {OUTPUT_DIR}/
""")