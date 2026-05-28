# Epithelial Map Clustering

Projeto de clustering para identificar perfis de mapas epiteliais com base nas regiões:

- C
- S
- ST
- T
- IT
- I
- IN
- N
- SN

As variáveis `Age`, `Gender` e `Eye` serão usadas apenas como complemento descritivo após a formação dos clusters.

## Estrutura
- `data/raw`: base original
- `data/processed`: saídas tratadas
- `src`: scripts principais
- `outputs/charts`: gráficos
- `outputs/tables`: tabelas
- `notebooks`: exploração
- `presentation`: roteiro e anotações

## Execução
```bash
py -3.11 -m pip install -r requirements.txt
py -3.11 src\test_clusters.py
py -3.11 src\main.py
py -3.11 src\generate_plots.py
```
