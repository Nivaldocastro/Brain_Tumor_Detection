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
## 📂 Dataset

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

Este projeto utiliza bibliotecas de visão computacional, processamento de imagens, extração de características e aprendizado de máquina para detecção de tumores cerebrais em imagens MRI.

---

**OpenCV (cv2):** Biblioteca de visão computacional utilizada para processamento de imagens.
Foi utilizada para leitura, redimensionamento, conversão de cores e pré-processamento das imagens MRI.
```python
import cv2
```
**OS:** Biblioteca nativa do Python utilizada para manipulação de arquivos e diretórios.
Permite navegar pelas pastas do dataset, carregar imagens e gerenciar caminhos de arquivos.
```python
import os
```
**NumPy:** Biblioteca fundamental para operações numéricas e manipulação de arrays.
Foi utilizada para processamento eficiente das imagens e operações matemáticas em matrizes.
```python
import numpy as np
```
**Pandas:** Biblioteca utilizada para manipulação e organização de dados.
Permite estruturar resultados em DataFrames e facilitar análises dos dados extraídos das imagens.
```python
import pandas as pd
```
**Matplotlib:** Biblioteca de visualização de dados em Python.
Foi utilizada para exibir imagens, gerar gráficos e visualizar resultados dos modelos.
```python
import matplotlib.pyplot as plt
```
**Collections (Counter):** Ferramenta para contagem de elementos em estruturas de dados.
Foi utilizada para analisar a distribuição das classes do dataset (com tumor e sem tumor).
```python
from collections import Counter
```
**Scikit-image (skimage):** Biblioteca para processamento e análise de imagens.

Foi utilizada para extração de características de textura das imagens MRI.
* graycomatrix: cálculo da matriz de coocorrência de níveis de cinza (GLCM)
* graycoprops: extração de propriedades estatísticas de textura
* local_binary_pattern: extração de padrões locais de textura (LBP)
```python
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
```
**Scikit-learn (sklearn):** Biblioteca principal de aprendizado de máquina.

Foi utilizada para divisão dos dados, pré-processamento, treinamento, otimização e avaliação dos modelos de classificação.

**train_test_split:** divisão dos dados em treino e teste

**GridSearchCV:** busca de melhores hiperparâmetros

**StratifiedKFold:** validação cruzada mantendo proporção das classes
```python
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
```

**StandardScaler:** normalização dos dados para que fiquem com média 0 e desvio padrão 1 
```python
from sklearn.preprocessing import StandardScaler
```

**Pipeline:** organização do fluxo de processamento e treinamento
```python
from sklearn.pipeline import Pipeline
```

**SVC (SVM):** Support Vector Machine

Algoritmo de classificação que encontra o melhor limite de separação entre classes.
É eficiente para problemas de alta dimensionalidade e foi utilizado para classificar imagens com e sem tumor.
```python
from sklearn.svm import SVC
```
**KNeighborsClassifier:** K-Nearest Neighbors

Algoritmo baseado em proximidade que classifica um dado com base nos seus vizinhos mais próximos.
A classe é definida pela maioria dos vizinhos semelhantes.
```python
from sklearn.neighbors import KNeighborsClassifier
```
**LogisticRegression:** regressão logística

Modelo estatístico utilizado para classificação binária.
Estima a probabilidade de uma imagem pertencer à classe com tumor ou sem tumor.
```python
from sklearn.linear_model import LogisticRegression
```
**RandomForestClassifier:** Random Forest

Algoritmo baseado em múltiplas árvores de decisão.
Combina vários modelos para melhorar a precisão e reduzir overfitting.
```python
from sklearn.ensemble import RandomForestClassifier
```
**classification_report:** métricas de avaliação do modelo

**ConfusionMatrixDisplay:** visualização da matriz de confusão
```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
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
