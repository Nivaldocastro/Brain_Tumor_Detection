# Projeto Detecção de tumores serebrais com machine learning

Este projeto tem como objetivo aplicar modelos como KNN (K-Nearest Neighbors ou K-Vizinhos Mais Próximos), SVM (Support Vector Machine), Logistic regression e Random Forest em um dataset sobre imagens de exames de ressonância magnética afim de treinalos e posteriormente análisar qaul modelo é melhor comparando não só a acuracy, mas também o desenpenho em relação aos predicts sobre as imagens com tumor.

---

## 📁 Estrutura do Projeto
```
├── preprocessamento.py            # Pré-processamento e correlação
|    ├──── X.npy, y.npy               # Armazenamento dos dados pré-processados 
├── split+data_mining.py           # Separação do treino e teste mais a extração de dados
|    ├──── brain_mri_train.csv        # Armazenamento dos dados coletados da extração
|    ├──── brain_mri_test.csv
├── classificacao.py               # Classificação com GridSearchCV
├── imagem                         # Imagens de resultados
└── README.md
```

---
📂 Dataset

Fonte: Kaggle

Link: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

Nome do dataset original: Brain MRI Images for Brain Tumor Detection

O dataset contém imagens de ressonância magnética (MRI) do cérebro utilizadas para classificação da presença de tumor cerebral.

Ele possui imagens organizadas em classes, incluindo:

presença de tumor cerebral (yes)

ausência de tumor cerebral (no)

As imagens são rotuladas e organizadas em diretórios separados, permitindo tarefas de classificação supervisionada em visão computacional.

A estrutura típica do dataset contém 253 imagens MRI do cérebro e inclui:

yes/ → 155 imagens com tumor cerebral

no/ → 98 imagens sem tumor cerebral

---

## Bibliotecas utilizadas

Este projeto foi desenvolvido em Python utilizando bibliotecas amplamente empregadas em análise de dados e aprendizado de máquina, conforme descrito abaixo:

---

**Pandas:** Biblioteca utilizada para carregamento, manipulação e análise de dados tabulares.
Permite ler arquivos CSV, tratar colunas, selecionar variáveis e realizar análises estatísticas básicas.

**Seaborn:** Biblioteca de visualização estatística baseada no matplotlib.
Facilita a criação de gráficos mais elegantes, como mapas de correlação, boxplots e distribuições.

**Matplotlib:** Biblioteca fundamental para criação de gráficos em Python.
Foi utilizada para plotar gráficos de dispersão, retas de regressão e gráficos de importância dos atributos.
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
**NumPy:** Biblioteca fundamental para operações numéricas e matemáticas em Python.
Foi utilizada para cálculos como o RMSE, manipulação de arrays e operações vetoriais.
```python
import numpy as np
```
**train_test_split:** Função do scikit-learn utilizada para dividir o dataset em conjuntos de treino e teste, garantindo uma avaliação adequada do modelo.
```python
from sklearn.model_selection import train_test_split
```
**StandardScaler:** Utilizada para padronização dos dados numéricos, fazendo com que todas as variáveis tenham média 0 e desvio padrão 1.
Essa etapa é essencial para modelos sensíveis à escala, como Ridge e Lasso.
```python
from sklearn.preprocessing import StandardScaler
```
**LinearRegression:** Modelo de Regressão Linear do scikit-learn.
Foi aplicado tanto na regressão linear simples quanto na regressão linear múltipla.
```python
from sklearn.linear_model import LinearRegression
```
**Ridge Regression:** Modelo de regressão linear com regularização L2, utilizado para reduzir overfitting e controlar a magnitude dos coeficientes.
```python
from sklearn.linear_model import Ridge
```
**Lasso Regression:** Modelo de regressão linear com regularização L1, capaz de zerar coeficientes, sendo útil para seleção de atributos e análise de importância das variáveis.
```python
from sklearn.linear_model import Lasso
```
**cross_val_score:** Função utilizada para aplicar validação cruzada (cross-validation), permitindo avaliar o desempenho dos modelos de forma mais robusta.
```python
from sklearn.model_selection import cross_val_score
```
**Métricas de Avaliação:** Foram utilizadas métricas para avaliar o desempenho dos modelos de regressão:
RMSE (Root Mean Squared Error): mede o erro médio das previsões.
R² (Coeficiente de Determinação): indica o quanto o modelo explica a variabilidade da variável alvo.
```python
from sklearn.metrics import mean_squared_error, r2_score
```

---

## Pré-processamento 

**Arquivo:** `preprocessamento.py`

Nesta etapa inicial, foi realizado o preparo dos dados para a modelagem:

---

## split + data_mining

**Arquivo:** `split+data_mining.py`

---

## Comparação: Linear vs Ridge vs Lasso

**Arquivo:** `linear_ridge_lasso.py`

---

## Coeficientes e Seleção de Atributos com Lasso

**Arquivo:** `coeficientes.py`

---

## Conclusão 

---

Projeto desenvolvido para fins acadêmicos e aprendizado em Machine Learning.
