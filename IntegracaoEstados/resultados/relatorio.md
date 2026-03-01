# Relatório: Análise de Clusterização da Integração Comercial dos Estados Brasileiros

Alunos: 

* André Machado Silva - 12221BSI201
* Thales Cavalheiro de Oliveira da Luz - 12311BSI265
* Pedro Biagi Dias - 12221BSI200

## Objetivo

Verificar, por meio de técnicas de clusterização, se os estados do Norte/Nordeste são mais integrados a economias emergentes, enquanto os estados do Sul/Sudeste são mais integrados a economias desenvolvidas.

---

## 1. Dados Utilizados

Foram utilizados dados de exportações brasileiras por UF e país de destino (2025), provenientes de quatro bases CSV organizadas por região de destino:

| Base                       | Registros         |
| -------------------------- | ----------------- |
| América do Norte / Europa | 129.649           |
| América do Sul            | 119.818           |
| Ásia                      | 65.691            |
| África e Oceania          | 96.668            |
| **Total unificado**  | **411.826** |

Após remoção de registros com UF "Não Declarada" e valores nulos, restaram **408.626 registros**. Os destinos abrangem **205 países** — 37 classificados como economias desenvolvidas e 168 como emergentes — distribuídos em 7 blocos econômicos: América do Norte, Europa, América do Sul, Ásia (exclusive Oriente Médio), Oriente Médio, África e Oceania.

## 2. Construção das Features

Para cada um dos **27 estados**, foram construídas **20 variáveis numéricas**:

1. Proporção (%) do valor exportado por bloco econômico (7 variáveis);
2. Proporção (%) para os 8 principais países parceiros: China, EUA, Argentina, Índia, Holanda, Chile, Japão e México;
3. Proporção (%) para economias desenvolvidas vs. emergentes (2 variáveis);
4. Número de países de destino (diversificação);
5. Valor total exportado (log-transformado);
6. Índice HHI de concentração de destinos. *HHI seria um indice de concentracao de mercado. Ele mede o quanto as exportacoes estáo concentradas em poucos países de destino.

## 3. Análise Exploratória

### 3.1 Destinos Dominantes

A Ásia (exclusive Oriente Médio) é o principal destino das exportações estaduais, com média de **37,1%** do valor, seguida pela Europa (**20,1%**) e América do Norte (**15,4%**). A China, isoladamente, responde por **23,4%** em média. No agregado, **62,5%** das exportações vão para economias emergentes e **37,5%** para desenvolvidas.

### 3.2 Heterogeneidade entre Estados

As features com maior coeficiente de variação (CV > 1) são: % Japão (CV = 2,37), % Argentina (1,79), % Oceania (1,50), % Oriente Médio (1,33), % Holanda (1,36), % Chile (1,25) e % América do Sul (1,14) — indicando grande disparidade entre os perfis exportadores estaduais.

### 3.3 Correlações Relevantes

Pares com correlação de Pearson |r| > 0,85:

| Par                                  | r       |
| ------------------------------------ | ------- |
| % Ásia × % China                   | 0,931   |
| % América do Norte × % EUA         | 0,882   |
| N_Países_Destino × Log_Valor_Total | 0,881   |
| % Desenvolvida × % Emergente        | −1,000 |

### 3.4 Outliers

Foram identificados outliers em 13 das 20 features. Destaques: Ceará (% América do Norte, % EUA, % México), Amapá (% Japão, % Índia, % Europa), Piauí (HHI), Roraima (% América do Sul). Os outliers foram mantidos, pois com apenas 27 observações cada estado é relevante e queríamos manter a lógica de todas observacoes e estados.

## 4. Redução de Dimensionalidade

### 4.1 PCA

| Componente | Variância Individual | Variância Acumulada |
| ---------- | --------------------- | -------------------- |
| PC1        | 30,93%                | 30,93%               |
| PC2        | 19,30%                | 50,23%               |
| PC3        | 12,61%                | 62,84%               |
| PC4        | 9,49%                 | 72,33%               |
| PC5        | 7,26%                 | 79,59%               |
| PC6        | 5,44%                 | 85,03%               |
| ...        | ...                   | ...                  |
| PC10       | 1,66%                 | 96,48%               |

São necessárias **10 componentes** para explicar ≥ 95% da variância.

**PC1** (30,9%) representa o eixo desenvolvida/emergente: loadings positivos elevados em % Desenvolvida (0,371), % EUA (0,335), % América do Norte (0,339) e negativos em % Emergente (−0,371), % Ásia (−0,318) e % China (−0,313).

**PC2** (19,3%) captura diversificação e comércio regional: loadings altos em % México (0,393), % Chile (0,387), % Argentina (0,377) e N_Países_Destino (0,350).

**PC3** (12,6%) reflete escala e orientação África/América do Sul: loadings altos em Log_Valor_Total (0,475), N_Países_Destino (0,393) e negativos em % América do Sul (−0,346) e % África (−0,333).

### 4.2 Espaço de Clustering

O clustering foi realizado sobre **6 componentes PCA** (85,0% da variância), reduzindo dimensionalidade e ruído.

### 4.3 t-SNE

Foi aplicado t-SNE (perplexity=13, 3000 iterações) para visualização 2D complementar.

## 5. Clusterização

### 5.1 Determinação do K Ótimo

| K | Inércia | Silhouette       | Pureza | Entropia (bits) |
| - | -------- | ---------------- | ------ | --------------- |
| 2 | 326,63   | **0,3420** | 0,5926 | 1,2806          |
| 3 | 261,60   | 0,2075           | 0,6667 | 0,8510          |
| 4 | 217,92   | 0,2278           | 0,5926 | 1,0677          |
| 5 | 171,10   | 0,2508           | 0,5926 | 1,0762          |
| 6 | 133,94   | 0,2927           | 0,6667 | 0,9485          |
| 7 | 106,55   | 0,2875           | 0,7037 | 0,8781          |
| 8 | 85,16    | 0,3014           | 0,7407 | 0,7263          |
| 9 | 69,74    | 0,3157           | 0,7037 | 0,7838          |

O **K = 2** apresenta o maior Silhouette Score (0,3420), sendo selecionado como ótimo.

### 5.2 K-Means (K = 2)

| Métrica         | Valor       |
| ---------------- | ----------- |
| Inércia (SSE)   | 326,63      |
| Silhouette Score | 0,3420      |
| Pureza           | 0,5926      |
| Entropia         | 1,2806 bits |

**Estabilidade:** 5 execuções com sementes diferentes (0, 1, 2, 42, 100) produziram partições idênticas — ARI médio = **1,0000** e NMI médio = **1,0000**. O resultado é perfeitamente estável.

### 5.3 Perfil dos Clusters

| Característica           | Cluster 0 (21 estados) | Cluster 1 (6 estados) |
| ------------------------- | ---------------------- | --------------------- |
| % Economias Emergentes    | **72,0%**        | 29,2%                 |
| % Economias Desenvolvidas | 28,0%                  | **70,8%**       |
| % China                   | **29,1%**        | 3,6%                  |
| % Estados Unidos          | 6,2%                   | **23,3%**       |
| % Europa                  | 15,6%                  | **36,0%**       |
| % Ásia                   | **42,7%**        | 17,2%                 |
| % América do Sul         | 14,2%                  | 5,5%                  |
| % África                 | 9,9%                   | 7,9%                  |
| N° Países Destino       | 121,9                  | 84,2                  |
| Log Valor Total           | 22,40                  | 20,46                 |
| HHI (concentração)      | 0,17                   | 0,13                  |

- **Cluster 0 — Orientação Emergente:** exportações concentradas em economias emergentes, especialmente China e Ásia. Maior volume total exportado e mais parceiros comerciais, porém maior concentração (HHI = 0,17).
- **Cluster 1 — Orientação Desenvolvida:** exportações direcionadas a EUA e Europa. Menor escala de exportação e menor diversificação de parceiros, mas menor concentração de destinos (HHI = 0,13).

### 5.4 Composição Regional dos Clusters

| Cluster          | Centro-Oeste | Nordeste | Norte | Sudeste | Sul |
| ---------------- | ------------ | -------- | ----- | ------- | --- |
| 0 (Emergente)    | 4            | 5        | 6     | 3       | 3   |
| 1 (Desenvolvido) | 0            | 4        | 1     | 1       | 0   |

Por macrorregião:

| Cluster          | Norte/Nordeste | Centro-Oeste | Sul/Sudeste |
| ---------------- | -------------- | ------------ | ----------- |
| 0 (Emergente)    | 11             | 4            | 6           |
| 1 (Desenvolvido) | 5              | 0            | 1           |

**Tabela dos estados por cluster:**

| Estado              | Região      | Cluster | % Desenvolvida | % Emergente | % China | % EUA |
| ------------------- | ------------ | ------- | -------------- | ----------- | ------- | ----- |
| Distrito Federal    | Centro-Oeste | 0       | 15,2           | 84,8        | 29,1    | 2,9   |
| Goiás              | Centro-Oeste | 0       | 24,7           | 75,3        | 43,3    | 4,8   |
| Mato Grosso         | Centro-Oeste | 0       | 15,8           | 84,2        | 41,0    | 0,9   |
| Mato Grosso do Sul  | Centro-Oeste | 0       | 21,9           | 78,1        | 44,8    | 5,0   |
| Bahia               | Nordeste     | 0       | 39,0           | 61,0        | 24,5    | 6,2   |
| Maranhão           | Nordeste     | 0       | 50,1           | 49,9        | 32,1    | 11,7  |
| Pernambuco          | Nordeste     | 0       | 24,7           | 75,3        | 3,0     | 4,6   |
| Piauí              | Nordeste     | 0       | 12,4           | 87,6        | 70,3    | 2,6   |
| Alagoas             | Nordeste     | 0       | 55,5           | 44,5        | 9,8     | 8,0   |
| Acre                | Norte        | 0       | 14,8           | 85,2        | 5,7     | 1,3   |
| Amazonas            | Norte        | 0       | 22,4           | 77,6        | 8,6     | 5,4   |
| Pará               | Norte        | 0       | 35,2           | 64,8        | 45,2    | 4,3   |
| Rondônia           | Norte        | 0       | 21,2           | 78,8        | 32,7    | 4,3   |
| Roraima             | Norte        | 0       | 8,1            | 91,9        | 26,8    | 0,2   |
| Tocantins           | Norte        | 0       | 19,3           | 80,7        | 55,0    | 1,8   |
| Minas Gerais        | Sudeste      | 0       | 41,1           | 58,9        | 34,9    | 9,3   |
| Rio de Janeiro      | Sudeste      | 0       | 43,6           | 56,4        | 37,6    | 13,5  |
| São Paulo          | Sudeste      | 0       | 38,4           | 61,6        | 13,8    | 18,8  |
| Paraná             | Sul          | 0       | 23,4           | 76,6        | 22,3    | 5,0   |
| Rio Grande do Sul   | Sul          | 0       | 26,2           | 73,8        | 21,5    | 7,3   |
| Santa Catarina      | Sul          | 0       | 35,6           | 64,4        | 9,6     | 11,8  |
| Ceará              | Nordeste     | 1       | 72,8           | 27,2        | 3,6     | 43,9  |
| Paraíba            | Nordeste     | 1       | 57,6           | 42,4        | 5,0     | 15,5  |
| Rio Grande do Norte | Nordeste     | 1       | 81,8           | 18,2        | 2,1     | 14,5  |
| Sergipe             | Nordeste     | 1       | 73,8           | 26,2        | 0,6     | 30,9  |
| Amapá              | Norte        | 1       | 77,7           | 22,3        | 4,5     | 8,7   |
| Espírito Santo     | Sudeste      | 1       | 61,1           | 38,9        | 5,7     | 26,4  |

Dos 6 estados no cluster voltado a economias desenvolvidas, **5 pertencem ao Norte/Nordeste** (Ceará, Paraíba, RN, Sergipe, Amapá) e apenas 1 ao Sudeste (Espírito Santo). Todos os grandes exportadores do Sul/Sudeste (SP, MG, RJ, PR, RS, SC) ficaram no cluster de orientação emergente.

### 5.5 Validação com Outros Algoritmos

- **Hierárquico (Ward, K=2):** Silhouette = 0,2822. Concordância parcial com o K-Means.
- **DBSCAN (eps=3,372, min_samples=2):** 3 clusters + 3 pontos de ruído, Silhouette = 0,3033. Confirma a estrutura de 2–3 grupos nos dados.

## 6. Resposta à Pergunta de Pesquisa

> *Estados do Norte/Nordeste são mais integrados a economias emergentes, enquanto Sul/Sudeste aos desenvolvidos?*

**A hipótese não se confirma.** Os resultados indicam que:

1. **A clusterização revela um padrão inverso ao esperado.** O cluster voltado a economias desenvolvidas (Cluster 1) é composto majoritariamente por estados do Nordeste (Ceará, Paraíba, RN, Sergipe) e Norte (Amapá), com exportações direcionadas a EUA e Europa. Enquanto isso, SP, MG, PR e RS — tipicamente associados a economias avançadas — estão no cluster de orientação emergente, pois exportam intensamente para a China.
2. **A pauta exportadora determina a integração, não a macrorregião.** Estados agroexportadores e mineradores (Mato Grosso com 41% para China, Piauí com 70%, Tocantins com 55%, Pará com 45%) dependem fortemente do mercado chinês independente da região geográfica. Já estados com pautas industriais ou de nicho (Ceará com 44% para EUA, Sergipe com 31%, Espírito Santo com 26%) formam um grupo distinto voltado a economias desenvolvidas.
3. **A China é o principal fator de diferenciação.** Com loading de −0,313 na PC1 e correlação de 0,931 com % Ásia, a dependência chinesa é a variável que mais separa os clusters. O Cluster 0 exporta em média 29,1% para a China; o Cluster 1, apenas 3,6%.

**Conclusão:** A integração comercial dos estados brasileiros é determinada pela **composição da pauta exportadora** (commodities agrícolas/minerais vs. manufaturados) e pela **dependência do mercado chinês**, e não pela dicotomia geográfica Norte/Nordeste vs. Sul/Sudeste.

## 7. Limitações

- Recorte temporal restrito (apenas dados de 2025), sem capturar dinâmicas ou tendências;
- Classificação desenvolvido/emergente simplificada (ex.: Coreia do Sul e Singapura como desenvolvidos);
- Análise baseada apenas em valor FOB de exportações, sem considerar investimentos ou serviços;
- Valor FOB pode ser distorcido por commodities de alto preço unitário (ex.: petróleo);
- Variáveis como Fluxo e URF apresentaram 76,5% de dados faltantes, não sendo utilizadas.
