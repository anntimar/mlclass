# 04_Eyes_Clustering

Projeto da atividade de clustering para identificar perfis de olhos com base **somente** nas variáveis:

- AL = comprimento axial do olho
- ACD = profundidade de câmara anterior
- WTW = distância branco a branco
- K1 = curvatura no meridiano menos curvo
- K2 = curvatura no meridiano mais curvo

## Regras da atividade
- Os grupos devem ser definidos **apenas** com `AL`, `ACD`, `WTW`, `K1` e `K2`.
- A coluna `Correto` **não** deve ser usada para criar os clusters.
- `Correto` pode ser usada **somente depois** para complementar a descrição dos grupos.

## Estrutura
- `data/raw`: base original
- `data/processed`: bases tratadas e resultados com cluster
- `notebooks`: notebook para exploração/análise
- `src`: scripts principais
- `outputs/tables`: tabelas finais
- `outputs/charts`: gráficos salvos
- `outputs/reports`: resumos do projeto
- `presentation`: roteiro e observações para apresentação

## Como rodar
1. Instale as dependências:
   ```bash
   py -3.11 -m pip install -r requirements.txt
   ```
2. Vá para a pasta `src`:
   ```bash
   cd src
   ```
3. Rode a análise:
   ```bash
   py -3.11 main.py
   ```

## O que o projeto faz
- Lê a base Excel original
- Seleciona apenas as variáveis do clustering
- Padroniza os dados
- Testa KMeans com 2, 3, 4 e 5 clusters
- Calcula silhouette score
- Escolhe o melhor cenário
- Gera tabela de perfil médio por cluster
- Gera gráficos para apresentação
- Mostra a frequência de `Correto` por cluster apenas como complemento
