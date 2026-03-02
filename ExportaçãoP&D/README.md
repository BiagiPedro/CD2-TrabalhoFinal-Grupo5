# Exportações de Alta Tecnologia e Investimento em P&D nos Estados Brasileiros

**Hipótese:** Existe correlação entre exportações de produtos de alta tecnologia e investimento estadual em P&D no Brasil? (2000–2023)

---

## 📁 Datasets necessários

Você precisará de **3 arquivos**, que devem ser baixados manualmente nas fontes abaixo:

| Arquivo | Fonte |
|---|---|
| `exportacoes_alta_tecnologia_uf.xlsx` | [ComexStat / MDIC](https://comexstat.mdic.gov.br/pt/geral) |
| `despesas_pd_uf_2000_2023.csv` | [MCTI – Tabela 2.3.5](https://www.gov.br/mcti/pt-br/acompanhe-o-mcti/indicadores/paginas/recursos-aplicados/governos-estaduais/2-3-5-brasil-dispendios-dos-governos-estaduais-em-pesquisa-e-desenvolvimento-por-execucao-segundo-regioes-e-unidades-da-federacao) |
| `percentual_pd_receitas_uf_2000_2023.csv` | [MCTI – Tabela 2.3.8](https://www.gov.br/mcti/pt-br/acompanhe-o-mcti/indicadores/paginas/recursos-aplicados/governos-estaduais/2-3-8-brasil-percentual-dos-dispendios-em-pesquisa-e-desenvolvimento-dos-governos-estaduais-em-relacao-as-suas-receitas-totais) |

---

## ☁️ Como configurar o Google Drive

O notebook roda no **Google Colab** e lê os arquivos diretamente do seu Google Drive.

1. Acesse [drive.google.com](https://drive.google.com)
2. Faça upload dos 3 arquivos acima para a **raiz do seu Drive** (`Meu Drive/`)
3. Certifique-se de que os nomes dos arquivos são exatamente os indicados na tabela acima

---

## 🚀 Como rodar o notebook

### Passo 1 — Abrir no Colab
- Acesse o repositório no GitHub
- Clique no arquivo `analiseExportacoesP_D.ipynb`
- Clique no botão **"Open in Colab"** (ou acesse [colab.research.google.com](https://colab.research.google.com) e faça upload manual do notebook)

### Passo 2 — Conectar o Google Drive
- Na primeira célula de código, o notebook executará:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- Uma janela pedirá sua permissão → clique em **"Permitir"** e faça login com sua conta Google

### Passo 3 — Executar tudo
- No menu superior, clique em **Runtime → Run all** (ou `Ctrl + F9`)
- Aguarde a execução completa — o notebook rodará automaticamente todas as etapas de análise

---

## 📊 O que o pipeline faz

O notebook executa as seguintes etapas em sequência:

1. **Coleta e qualidade dos dados** — carrega e valida os 3 datasets
2. **Limpeza e transformação** — padroniza nomes de estados, trata valores ausentes
3. **Análise exploratória** — evolução temporal, ranking por estado, distribuição regional
4. **Correlação** — testa a relação estatística entre exportações e investimento em P&D
5. **Clusterização** — agrupa estados por perfil usando K-Means e Clusterização Hierárquica
6. **Visualizações interativas** — gráficos com Plotly para exploração dos resultados

---

## 🛠️ Dependências

Todas as bibliotecas já estão disponíveis no ambiente do Google Colab. Nenhuma instalação adicional é necessária.

Bibliotecas utilizadas: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `plotly`

---

