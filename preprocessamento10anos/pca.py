import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.preprocessing import StandardScaler

# ==============================
# 1️⃣ CARREGAMENTO DOS DADOS
# ==============================
path = "preprocessamento10anos/base10anosprocessadasemnormalizar.csv"

df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)

# ==============================
# 2️⃣ PREPARAÇÃO PARA REDUÇÃO DE DIMENSIONALIDADE
# ==============================
# Selecionamos as variáveis numéricas e binárias que definem o comportamento dos dados
cols_para_reducao = [
    "Valor US$ FOB", 
    "Quilograma Líquido", 
    "Produto_Estrategico", 
    "Periodo_Guerra",
    "Fluxo"
]

# Removendo possíveis NaNs apenas para a visualização
df_visualizacao = df[cols_para_reducao].dropna()

# A normalização é OBRIGATÓRIA para UMAP/t-SNE
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df_visualizacao)

# ==============================
# 3️⃣ APLICAÇÃO DO UMAP (Visualização 2D)
# ==============================
# n_neighbors: equilibra estrutura local vs global
# min_dist: controla o quão próximos os pontos ficam no gráfico
reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(dados_normalizados)

# Adicionando os componentes de volta ao dataframe para plotagem
df_visualizacao['umap_1'] = embedding[:, 0]
df_visualizacao['umap_2'] = embedding[:, 1]

# ==============================
# 4️⃣ VISUALIZAÇÃO DOS RESULTADOS
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Gráfico A: Colorido por Período de Guerra
sns.scatterplot(
    data=df_visualizacao, x='umap_1', y='umap_2', 
    hue='Periodo_Guerra', palette='coolwarm', ax=axes[0], alpha=0.6
)
axes[0].set_title('UMAP: Dispersão por Período de Guerra (0=Pré, 1=Pós)', fontsize=14)

# Gráfico B: Colorido por Produto Estratégico
sns.scatterplot(
    data=df_visualizacao, x='umap_1', y='umap_2', 
    hue='Produto_Estrategico', palette='viridis', ax=axes[1], alpha=0.6
)
axes[1].set_title('UMAP: Dispersão por Produto (0=Trigo, 1=Fertilizante)', fontsize=14)

plt.tight_layout()
plt.show()

# ==============================
# 5️⃣ EXPORTAÇÃO DOS COMPONENTES (Opcional)
# ==============================
# Se quiser usar as coordenadas X e Y no seu relatório final
print("Coordenadas UMAP geradas com sucesso.")
print(df_visualizacao[['umap_1', 'umap_2']].head())