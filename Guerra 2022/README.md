# CD2-TrabalhoFinal-Grupo5
Tese Estados com pauta similar à Rússia/Ucrânia (trigo, fertilizantes) foram mais afetados pela guerra de 2022

# Projeto de Clusterização

## 📂 Estrutura do projeto e execução

Para executar o projeto, basta verificar a organização das pastas e arquivos.

A primeira etapa corresponde ao **pré-processamento dos dados**, localizado na pasta:

preprocessamento10anos/


Nessa pasta estão os arquivos responsáveis por:
- Limpeza e preparação da base de dados
- Geração dos arquivos processados
- Execução da **Análise de Componentes Principais (PCA)**
- Geração das informações auxiliares para a análise

Após o pré-processamento, é possível executar os métodos de agrupamento presentes nas seguintes pastas:

- `k-mens/` → contém o código para execução do algoritmo **K-Means**
- `DBscan/` → contém o código para execução do algoritmo **DBSCAN**

Além disso, existe um arquivo específico voltado apenas para o cálculo dos índices necessários para a definição do parâmetro **eps** do DBSCAN:

- `CalculoEPS.py`

## ▶️ Execução

1. Execute primeiro os scripts da pasta `preprocessamento10anos/`.
2. Em seguida, execute os scripts de:
   - `k-mens/` para K-Means
   - `DBscan/` para DBSCAN
3. Caso deseje calcular apenas os índices para definição do parâmetro `eps`, execute:
   - `CalculoEPS.py`
