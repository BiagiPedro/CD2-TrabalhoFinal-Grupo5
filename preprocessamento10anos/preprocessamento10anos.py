
import pandas as pd
import unicodedata
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==============================
# 1️⃣ LEITURA
# ==============================

df = pd.read_csv(
    "preprocessamento10anos/importação10anoscomestado.csv",
    sep=";",
    encoding="utf-8-sig",
    low_memory=False
)

# ==============================
# 2️⃣ LIMPEZA TEXTO
# ==============================

def limpar_texto(texto):
    if isinstance(texto, str):
        texto = texto.replace("\r", "")
        texto = texto.strip()
        texto = texto.upper()
        texto = unicodedata.normalize('NFKD', texto)\
                 .encode('ASCII', 'ignore')\
                 .decode('ASCII')
    return texto

colunas_texto = df.select_dtypes(include=["object", "string"]).columns
for col in colunas_texto:
    df[col] = df[col].apply(limpar_texto)

# ==============================
# 3️⃣ CRIAR FLAG DE PRODUTO E REMOVER COLUNAS
# ==============================

# Garantir que NCM seja string
df["Código NCM"] = df["Código NCM"].astype(str).str.strip()

# Criar flag estratégica
df["Produto_Estrategico"] = np.where(
    df["Código NCM"].str.startswith("1001") |  # Trigo
    df["Código NCM"].str.startswith("31"),    # Fertilizantes
    1,
    0
).astype("int8")

# Remover colunas desnecessárias
df.drop(columns=["Descrição NCM", "Código NCM","URF"], inplace=True)

# ==============================
# 4️⃣ MÊS → MANTER COMO STRING
# ==============================

df["Mês"] = (
    df["Mês"]
    .astype(str)
    .str.split(".")
    .str[1]
    .str.strip()
)

# ==============================
# 5️⃣ CRIAR BOOLEANOS (mais performático)
# ==============================

# Fluxo: EXPORTACAO = 1, IMPORTACAO = 0
df["Fluxo"] = df["Fluxo"].map({
    "EXPORTACAO":1,
    "IMPORTACAO":0
}).astype("int8")

# Mapa de meses (apenas para cálculo interno)
mes_ordem = {
    "JANEIRO":1,"FEVEREIRO":2,"MARCO":3,"ABRIL":4,"MAIO":5,
    "JUNHO":6,"JULHO":7,"AGOSTO":8,"SETEMBRO":9,
    "OUTUBRO":10,"NOVEMBRO":11,"DEZEMBRO":12
}

mes_temp = df["Mês"].map(mes_ordem)

# Período guerra (0 = pré, 1 = pós)
df["Periodo_Guerra"] = np.where(
    (df["Ano"] < 2022) | ((df["Ano"] == 2022) & (mes_temp < 2)),
    0,
    1
).astype("int8")


# ==============================
# 7️⃣ TRATAMENTO DE OUTLIERS (IQR)
# ==============================

colunas_outliers = ["Valor US$ FOB", "Quilograma Líquido"]

for col in colunas_outliers:
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Winsorização (capping)
    df[col] = np.where(df[col] < limite_inferior, limite_inferior, df[col])
    df[col] = np.where(df[col] > limite_superior, limite_superior, df[col])

print("Outliers tratados via IQR.")

print("Shape final:", df.shape)
print(df.info())
print(df.head())

# # ==============================
# # 8️⃣ NORMALIZAÇÃO (in-place)
# # ==============================

# from sklearn.preprocessing import StandardScaler

# # Garantir que são numéricas
# df["Valor US$ FOB"] = pd.to_numeric(df["Valor US$ FOB"], errors="coerce")
# df["Quilograma Líquido"] = pd.to_numeric(df["Quilograma Líquido"], errors="coerce")

# # Selecionar colunas contínuas
# colunas_normalizar = [
#     "Valor US$ FOB",
#     "Quilograma Líquido",
# ]

# scaler = StandardScaler()

# df[colunas_normalizar] = scaler.fit_transform(df[colunas_normalizar])
# print("Shape final:", df.shape)
# print(df.info())
# print(df.head())


# # # # # # Caminho onde será salvo
# caminho_saida = "base10anosprocessadasemnormalizar.csv"

# # # # # Salvar
# df.to_csv(caminho_saida, sep=";", index=False)

# print(f"Base salva com sucesso em: {caminho_saida}")



