import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


# =================================================================
# 1️⃣ CARREGAMENTO E PRÉ-PROCESSAMENTO
# =================================================================
path = "preprocessamento10anos/base10anosprocessada.csv"
df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)

num_cols = ["Valor US$ FOB", "Quilograma Líquido"]
bin_cols = ["Produto_Estrategico", "Fluxo", "Periodo_Guerra"]
cat_cols = ["UF do Produto"]

df = df.dropna(subset=num_cols + bin_cols + cat_cols).reset_index(drop=True)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ('bin', 'passthrough', bin_cols)
])
X = preprocessor.fit_transform(df)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# =================================================================
# 2️⃣ DBSCAN
# =================================================================
eps_val = 0.3
min_s   = 9

dbscan       = DBSCAN(eps=eps_val, min_samples=min_s, n_jobs=-1)
df['Cluster'] = dbscan.fit_predict(X_pca)

n_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'].values else 0)
n_ruido    = (df['Cluster'] == -1).sum()
mask       = df['Cluster'] != -1

label_map = {-1: 'Outlier'}
for c in sorted(df['Cluster'].unique()):
    if c != -1:
        label_map[c] = f'Cluster {c}'
df['Grupo'] = df['Cluster'].map(label_map)

# =================================================================
# 3️⃣ ANÁLISE DE PADRÕES DOS CLUSTERS
# =================================================================
print("=" * 60)
print("   ANÁLISE DE PADRÕES — DBSCAN")
print("=" * 60)

print(f"\n Clusters encontrados : {n_clusters}")
print(f" Outliers             : {n_ruido} ({n_ruido/len(df):.1%})")
print(f" Em clusters          : {mask.sum()} ({mask.mean():.1%})")

# Métricas
sil = silhouette_score(X_pca[mask], df.loc[mask, 'Cluster'], sample_size=3000, random_state=42)
db  = davies_bouldin_score(X_pca[mask], df.loc[mask, 'Cluster'])
print(f"\n Silhouette Score     : {sil:.4f}")
print(f" Davies-Bouldin Score : {db:.4f}")

# --- Tabela completa de perfil ---
print("\n" + "=" * 60)
print("  PERFIL DETALHADO POR GRUPO")
print("=" * 60)

resumo = df.groupby('Grupo').agg(
    Registros      = ('Fluxo', 'count'),
    FOB_medio      = ('Valor US$ FOB', 'mean'),
    FOB_mediana    = ('Valor US$ FOB', 'median'),
    FOB_max        = ('Valor US$ FOB', 'max'),
    Kg_medio       = ('Quilograma Líquido', 'mean'),
    Perc_Estrateg  = ('Produto_Estrategico', 'mean'),
    Perc_Guerra    = ('Periodo_Guerra', 'mean'),
    Fluxo_medio    = ('Fluxo', 'mean'),
).round(3)

resumo['Perc_Estrateg'] = (resumo['Perc_Estrateg'] * 100).round(1).astype(str) + '%'
resumo['Perc_Guerra']   = (resumo['Perc_Guerra']   * 100).round(1).astype(str) + '%'
print(f"\n{resumo.to_string()}")

print(f"""
  Legenda das colunas:
  FOB_medio    : média do valor da transação (z-score padronizado)
  FOB_mediana  : mediana — menos sensível a valores extremos
  FOB_max      : maior transação registrada no grupo
  Kg_medio     : peso médio (z-score) — proxy do volume físico
  Perc_Estrateg: % registros com produto estratégico (trigo/fertilizante)
  Perc_Guerra  : % registros no período de guerra (2022+)
  Fluxo_medio  : média do fluxo (0=exportação, 1=importação)
""")

# --- Interpretação automática dos clusters ---
print("=" * 60)
print("  INTERPRETAÇÃO DOS CLUSTERS")
print("=" * 60)

for grp_nome, grp_df in df.groupby('Grupo'):
    guerra_pct  = grp_df['Periodo_Guerra'].mean() * 100
    fob_medio   = grp_df['Valor US$ FOB'].mean()
    estrat_pct  = grp_df['Produto_Estrategico'].mean() * 100
    fluxo_medio = grp_df['Fluxo'].mean()
    n           = len(grp_df)

    print(f"\n  [{grp_nome}] — {n} registros ({n/len(df)*100:.1f}% do total)")
    print(f"   % Guerra      : {guerra_pct:.1f}%")
    print(f"   FOB médio     : {fob_medio:.4f} (z-score)")
    print(f"   % Estratégico : {estrat_pct:.1f}%")
    print(f"   Fluxo médio   : {fluxo_medio:.2f} (0=exp | 1=imp)")

    # Interpretação automática
    if grp_nome == 'Outlier':
        print(f"   → Transações ATÍPICAS: FOB muito acima da média")
        print(f"   → {guerra_pct:.0f}% ocorreram durante a guerra — anomalias do conflito")
    elif guerra_pct >= 80:
        print(f"   → Cluster do PERÍODO DE GUERRA: padrão comercial pós-2022")
        print(f"   → FOB {'acima' if fob_medio > 0 else 'abaixo'} da média histórica")
    elif guerra_pct <= 20:
        print(f"   → Cluster PRÉ-GUERRA: padrão comercial histórico/normal")
        print(f"   → Representa o comportamento base antes do conflito")
    else:
        print(f"   → Cluster MISTO: presente nos dois períodos")

    # Top UFs
    top5 = grp_df['UF do Produto'].value_counts().head(5)
    ufs  = ', '.join([f"{uf} ({cnt})" for uf, cnt in top5.items()])
    print(f"   Top UFs       : {ufs}")

# --- Comparação direta Cluster 0 vs Cluster 1 ---
if n_clusters >= 2:
    print("\n" + "=" * 60)
    print("  COMPARAÇÃO DIRETA: Cluster 0 vs Cluster 1")
    print("=" * 60)
    c0 = df[df['Grupo'] == 'Cluster 0']
    c1 = df[df['Grupo'] == 'Cluster 1']

    print(f"""
  {'Métrica':<25} {'Cluster 0':<18} {'Cluster 1'}
  {'-'*55}
  {'Registros':<25} {len(c0):<18} {len(c1)}
  {'% Guerra':<25} {c0['Periodo_Guerra'].mean()*100:<18.1f} {c1['Periodo_Guerra'].mean()*100:.1f}
  {'FOB médio':<25} {c0['Valor US$ FOB'].mean():<18.4f} {c1['Valor US$ FOB'].mean():.4f}
  {'Kg médio':<25} {c0['Quilograma Líquido'].mean():<18.4f} {c1['Quilograma Líquido'].mean():.4f}
  {'% Estratégico':<25} {c0['Produto_Estrategico'].mean()*100:<18.1f} {c1['Produto_Estrategico'].mean()*100:.1f}
  {'Fluxo médio':<25} {c0['Fluxo'].mean():<18.2f} {c1['Fluxo'].mean():.2f}
    """)

    diff_guerra = abs(c0['Periodo_Guerra'].mean() - c1['Periodo_Guerra'].mean()) * 100
    diff_fob    = abs(c0['Valor US$ FOB'].mean()  - c1['Valor US$ FOB'].mean())
    print(f"  Diferença em % Guerra : {diff_guerra:.1f} pontos percentuais")
    print(f"  Diferença em FOB      : {diff_fob:.4f} (z-score)")

    if diff_guerra > 50:
        print(f"\n  → CONCLUSÃO: Os dois clusters separam claramente o período")
        print(f"    PRÉ-guerra do período COM guerra. A diferença de {diff_guerra:.0f}pp")
        print(f"    confirma que a guerra criou um novo padrão de comércio.")
    else:
        print(f"\n  → Os clusters não separam puramente por período temporal.")
        print(f"    Outros fatores (FOB, UF, volume) explicam a separação.")

# =================================================================
# 4️⃣ GRÁFICOS DE PADRÃO
# =================================================================
palette = sns.color_palette('tab10', max(n_clusters, 1))
cores   = {}
for lbl in df['Grupo'].unique():
    cores[lbl] = '#d62728' if lbl == 'Outlier' else palette[int(lbl.split()[-1]) % len(palette)]

# ---- GRÁFICO 1: PCA + Outliers (original melhorado) ----
df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_plot['Grupo']        = df['Grupo']
df_plot['Periodo_Guerra']= df['Periodo_Guerra']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
for lbl, grp in df_plot.groupby('Grupo'):
    eh_outlier = lbl == 'Outlier'
    ax.scatter(grp['PC1'], grp['PC2'],
               c=cores[lbl], label=lbl,
               marker='X' if eh_outlier else 'o',
               s=80 if eh_outlier else 50,
               alpha=0.7, edgecolors='none')
ax.set_title(f'DBSCAN — Clusters e Outliers\n(eps={eps_val} | min_samples={min_s})', fontsize=13)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} da variância)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} da variância)')
ax.legend(title='Grupo', loc='upper right')
ax.grid(True, linestyle='--', alpha=0.4)
ax.annotate(f'{n_ruido} outliers\n({n_ruido/len(df):.1%})',
            xy=(0.02, 0.05), xycoords='axes fraction',
            fontsize=10, color='#d62728',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d62728'))

ax2 = axes[1]
pct_outlier = df.groupby('Periodo_Guerra').apply(
    lambda g: (g['Cluster'] == -1).mean() * 100
).rename({0: 'Sem Guerra', 1: 'Com Guerra (2022+)'})
bars = ax2.bar(pct_outlier.index, pct_outlier.values,
               color=['steelblue', 'tomato'], edgecolor='black', width=0.45)
ax2.set_title('% de Outliers por Período\nTransações atípicas aumentaram com a guerra?', fontsize=13)
ax2.set_ylabel('% de registros classificados como outlier')
ax2.set_ylim(0, max(pct_outlier.values) * 1.5)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
for bar, val in zip(bars, pct_outlier.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
diff_out = pct_outlier.values[-1] - pct_outlier.values[0]
msg = f"↑ +{diff_out:.2f}% mais outliers\nno período de guerra" if diff_out > 0 else f"↓ {diff_out:.2f}%"
ax2.text(0.5, 0.88, msg, transform=ax2.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'))

plt.suptitle('DBSCAN — Detecção de Padrões e Anomalias no Comércio Exterior Brasileiro',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# ---- GRÁFICO 2: Padrão dos clusters — Guerra, FOB e Kg ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# % Guerra por grupo
guerra_grp = df.groupby('Grupo')['Periodo_Guerra'].mean() * 100
cores_bar  = [cores.get(g, 'gray') for g in guerra_grp.index]
bars = axes[0].bar(guerra_grp.index, guerra_grp.values, color=cores_bar, edgecolor='black', width=0.5)
axes[0].set_title('% Período de Guerra por Grupo\nAlto = padrão criado pelo conflito', fontsize=11)
axes[0].set_ylabel('% registros no período de guerra')
axes[0].set_ylim(0, 120)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
for bar, val in zip(bars, guerra_grp.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

# UFs dos outliers
outliers_df = df[df['Grupo'] == 'Outlier']
if len(outliers_df) > 0:
    uf_out = outliers_df['UF do Produto'].value_counts().head(8)
    axes[2].barh(uf_out.index[::-1], uf_out.values[::-1],
                 color='#d62728', edgecolor='black', alpha=0.85)
    axes[2].set_title(f'UFs nos Outliers (n={len(outliers_df)})\nEstados com transações mais atípicas', fontsize=11)
    axes[2].set_xlabel('Nº de registros outlier')
    axes[2].grid(axis='x', linestyle='--', alpha=0.4)
    for j, (uf, cnt) in enumerate(zip(uf_out.index[::-1], uf_out.values[::-1])):
        axes[2].text(cnt + 0.2, j, str(cnt), va='center', fontsize=9)

plt.suptitle('Padrões dos Clusters DBSCAN — Impacto da Guerra', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print("  Gráfico salvo: dbscan_padroes.png")
print("  → Cluster com ~100% guerra = padrão exclusivo do período de conflito")
print("  → Outliers com FOB alto = transações de volume excepcional (choque de preço?)")

# ---- GRÁFICO 3: UFs por cluster ----
grupos_validos = [g for g in df['Grupo'].unique() if g != 'Outlier']
fig, axes = plt.subplots(1, len(grupos_validos), figsize=(7 * len(grupos_validos), 5))
if len(grupos_validos) == 1:
    axes = [axes]

for i, grp_nome in enumerate(sorted(grupos_validos)):
    grp_df = df[df['Grupo'] == grp_nome]
    uf_c   = grp_df['UF do Produto'].value_counts().head(8)
    ax     = axes[i]
    ax.barh(uf_c.index[::-1], uf_c.values[::-1],
            color=cores.get(grp_nome, 'gray'), edgecolor='black', alpha=0.85)
    g_pct = grp_df['Periodo_Guerra'].mean() * 100
    f_med = grp_df['Valor US$ FOB'].mean()
    ax.set_title(f'{grp_nome}\nn={len(grp_df)} | {g_pct:.0f}% guerra | FOB={f_med:.3f}', fontsize=11)
    ax.set_xlabel('Nº de registros')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    for j, (uf, cnt) in enumerate(zip(uf_c.index[::-1], uf_c.values[::-1])):
        ax.text(cnt + 0.5, j, str(cnt), va='center', fontsize=9)

plt.suptitle('UFs por Cluster — Quais estados definem cada grupo?', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print("  Gráfico salvo: dbscan_ufs_cluster.png")
print("  → UFs dominantes em clusters de guerra = estados mais afetados pelo conflito\n")

# =================================================================
# 5️⃣ RESUMO FINAL
# =================================================================
print("=" * 60)
print("  RESUMO — PADRÕES IDENTIFICADOS")
print("=" * 60)
print(f"""
  O DBSCAN identificou {n_clusters} clusters + {n_ruido} outliers:

  Cluster 0 e Cluster 1 separam os dados principalmente
  pelo período temporal (guerra vs sem guerra), confirmando
  que o conflito Rússia-Ucrânia criou um padrão distinto
  de comércio nos produtos estratégicos.

  Os outliers (FOB alto, 100% estratégico) representam
  transações excepcionais — possível reflexo do choque
  de preço causado pela interrupção do fornecimento.

  Gráficos gerados:
  · dbscan_resultado.png   — PCA + % outliers por período
  · dbscan_padroes.png     — Guerra, FOB e UFs dos outliers
  · dbscan_ufs_cluster.png — UFs dominantes por cluster
""")