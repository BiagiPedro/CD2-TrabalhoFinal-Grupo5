import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1️⃣ CARREGAMENTO DOS DADOS
# ==============================
path = "preprocessamento10anos/base10anosprocessadasemnormalizar.csv"

df = pd.read_csv(
    path, 
    sep=";", 
    encoding="utf-8-sig", 
    low_memory=False
)

# Criando legenda para o gráfico
df['Status_Guerra'] = df['Periodo_Guerra'].map({0: 'Pré-Guerra', 1: 'Pós/Durante Guerra'})

# ==============================
# 2️⃣ GERAÇÃO DOS BOXPLOTS UNIFICADOS (Sem Avisos)
# ==============================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot Valor US$ FOB
# Ajuste: adicionado hue='Status_Guerra' e legend=False para remover o FutureWarning
sns.boxplot(
    x='Status_Guerra', 
    y='Valor US$ FOB', 
    data=df, 
    hue='Status_Guerra', 
    palette='coolwarm', 
    ax=axes[0], 
    legend=False
)
axes[0].set_title('Distribuição Geral: Valor US$ FOB', fontsize=14)
axes[0].set_ylabel('Valor (Normalizado)')

# Boxplot Quilograma Líquido
sns.boxplot(
    x='Status_Guerra', 
    y='Quilograma Líquido', 
    data=df, 
    hue='Status_Guerra', 
    palette='viridis', 
    ax=axes[1], 
    legend=False
)
axes[1].set_title('Distribuição Geral: Quilograma Líquido', fontsize=14)
axes[1].set_ylabel('Peso (Normalizado)')

plt.tight_layout()
################plt.show()

# ==============================
# 3️⃣ HEATMAP DE CORRELAÇÃO
# ==============================
plt.figure(figsize=(10, 8))

# Selecionamos apenas as colunas numéricas e binárias para a correlação
cols_corr = ["Valor US$ FOB", "Quilograma Líquido", "Produto_Estrategico", "Periodo_Guerra"]
corr_matrix = df[cols_corr].corr()

# Plotando o Heatmap
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='RdBu_r', 
    center=0, 
    fmt=".2f", 
    linewidths=0.5,
    annot_kws={"size": 12}
)
plt.title('Matriz de Correlação: Impacto da Guerra e Variáveis Comerciais', fontsize=15)
############plt.show()


# =================================================================
# 1️⃣ SEGMENTAÇÃO POR FLUXO (BOXPLOT CRUZADO)
# =================================================================
# Traduzindo Fluxo para facilitar a leitura (Assumindo 0/1 ou E/I)
# Ajuste o mapeamento conforme sua base (ex: 0=Exportação, 1=Importação)
plt.figure(figsize=(8,6))

sns.countplot(
    data=df,
    x='Status_Guerra',
    hue='Fluxo'
)

plt.title('Quantidade de Operações por Período de Guerra')
plt.ylabel('Número de Operações')
plt.xlabel('Período')
plt.legend(title='Fluxo (0=Export, 1=Import)')
###########plt.show()


#==========================================================================
# Configuração visual do gráfico
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Gerando o gráfico de contagem (representatividade)
grafico = sns.countplot(
    data=df, 
    x='Produto_Estrategico', 
    palette='viridis'
)

# Adicionando os valores exatos em cima de cada barra para facilitar a leitura
for p in grafico.patches:
    grafico.annotate(
        f'{int(p.get_height())}', 
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', va='baseline', 
        fontsize=12, color='black', 
        xytext=(0, 5), 
        textcoords='offset points'
    )

# Customização de títulos e legendas
plt.title('Representatividade da Variável: Produto Estratégico', fontsize=15)
plt.xlabel('Classe (0 = Outros, 1 = Trigo/Fertilizantes)', fontsize=12)
plt.ylabel('Quantidade de Registros (Frequência)', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['Outros (0)', 'Estratégico (1)'])

#####plt.show()


#==================================================
#   ESTADOS 
#==================================================

df['Desc_Fluxo'] = df['Fluxo'].map({0: 'Exportação', 1: 'Importação'})

plt.figure(figsize=(14,6))

sns.countplot(
    data=df,
    x='UF do Produto',
    hue='Desc_Fluxo'
)

plt.title('Quantidade de Operações por Estado e Fluxo')
plt.ylabel('Número de Operações')
plt.xlabel('Estado (UF)')
plt.xticks(rotation=90)

plt.legend(title='Fluxo')
plt.tight_layout()
plt.show()


# ==========================================================
# DISTRIBUIÇÃO DOS DADOS POR ANO
# ==========================================================

print("\n" + "="*60)
print(" DISTRIBUIÇÃO DOS DADOS POR ANO ")
print("="*60)

# Garantir que Ano é numérico
df['Ano'] = pd.to_numeric(df['Ano'], errors='coerce')

# Agrupamento por ano
distrib_ano = df.groupby('Ano').agg(
    Registros=('Fluxo', 'count'),
    Valor_Total_FOB=('Valor US$ FOB', 'sum'),
    Peso_Total_KG=('Quilograma Líquido', 'sum'),
    Perc_Estrategico=('Produto_Estrategico', 'mean'),
    Perc_Guerra=('Periodo_Guerra', 'mean')
).round(3)

# Transformar percentuais
distrib_ano['Perc_Estrategico'] = (distrib_ano['Perc_Estrategico'] * 100).round(1).astype(str) + '%'
distrib_ano['Perc_Guerra'] = (distrib_ano['Perc_Guerra'] * 100).round(1).astype(str) + '%'

# Ordenar por ano
distrib_ano = distrib_ano.sort_index()

print("\nResumo por Ano:")
print(distrib_ano.to_string())

# ==========================================================
# GRÁFICO 1 — Quantidade de Registros por Ano
# ==========================================================

plt.figure(figsize=(10,6))
df.groupby('Ano').size().plot(kind='bar')
plt.title('Quantidade de Registros por Ano')
plt.ylabel('Quantidade')
plt.xlabel('Ano')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================================
# GRÁFICO 2 — Valor Total FOB por Ano
# ==========================================================

plt.figure(figsize=(10,6))
df.groupby('Ano')['Valor US$ FOB'].sum().plot(kind='bar')
plt.title('Valor Total FOB por Ano')
plt.ylabel('Valor Total (US$)')
plt.xlabel('Ano')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nAnálise rápida:")
print(f"""
• Ano com mais registros: {df.groupby('Ano').size().idxmax()}
• Ano com maior valor FOB: {df.groupby('Ano')['Valor US$ FOB'].sum().idxmax()}
""")






