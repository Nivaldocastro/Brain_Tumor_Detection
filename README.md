# Projeto Detecção de tumores serebrais com machine learning

Este projeto tem como objetivo aplicar modelos como KNN (K-Nearest Neighbors ou K-Vizinhos Mais Próximos), SVM (Support Vector Machine), Logistic regression e Random Forest em um dataset sobre imagens de exames de ressonância magnética afim de treinalos e posteriormente análisar qaul modelo é melhor comparando não só a acuracy, mas também o desenpenho em relação aos predicts sobre as imagens com tumor.

---

## 📁 Estrutura do Projeto
```
├── preprocessamento.py            # Pré-processamento e correlação
|    ├──── X.npy, y.npy            # Armazenamento dos dados pré-processados 
├── split+data_mining.py           # Separação do treino e teste mais a extração de dados
|    ├──── brain_mri_train.csv     # Armazenamento dos dados coletados da extração
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

Nesta etapa inicial, foi realizado a importação do dataset, compreenção dos dados e ajustes para realizar a classificação

Ao importar o caminho do dataset é realizado primeiramente duas listas vazias, uma para armazenar as imagens e a outra para armazenar a label das imagens 

```
X = [] # vai guardar as imagens
y = [] # vai guardar a label de cada imagem
```
Posteriormente foi definida uma estrutura chamada `resize_com_padding`. O objetivo dela é redmencionar as imagens em um tamanho padrão 224x224 sem causar distorção.
Sobre o tamanho, foi analisado a média do tamanho das imagens e o resultado deu próximo do padrão 224x224.

Após fazer o `resize_com_padding`, foi aniciada o pré-processamento fazendo etapas como:
* implementar a função COLOR_BGR2GRAY para deixar as imagens apenas em tom de cinza
* verificação da dimenção das imagens
* Aplicadção do `resize_com_padding`
* normalização para escala de 0 a 1
* transformar a coluna labels yes = 1 e no = 0

Para validar esse pré-processamento foi feito um código usando o matplotlib
```python
idx = 200

plt.imshow(X[idx].reshape(224, 224), cmap="gray")
plt.title("Tumor" if y[idx] == 1 else "Sem tumor")
plt.axis("off")
plt.show()
```
O objetivo é analisar se as imagens não foram distorcidas, não foram cortadas, não perderam qualidade, ou sea, mantém detalhes importantes da imagem sem custo computacional muito alto.
Exemplo:

!(validação cérebro)[]

E após concluir a validação, os dados foram salvos para posteriormente ser implementado o split e realizado a mineração dos dados

---

## split + data_mining

**Arquivo:** `split+data_mining.py`

Nessa etapa, foi implementada o split (Separação do treino e test) e miretação de dados

**split**

O dataset foi dividido em 80% treino (202 dados) e 20% teste (51 dados) com 2 detalhes importantes
* Foi utilizando `random_state=42` para fazer com que os dados sejam separados de forma aleatória
* Foi utilizado o `stratify` para separar as classes de forma que preserve a estrutura original do dataset, ou seja, 60% com tumor e 40% sem tumor tanto no treino, tanto no teste.

Portando, ficou
```
Distribuição das classes no treino: [ 78 124]
Distribuição das classes no teste:  [ 20  31]
```
**data_mining**

Essa etapa realiza a mineração de dados das imagens MRI, transformando cada imagem em um conjunto de características numéricas (features) que podem ser utilizadas pelos modelos de machine learning para classificação.

Como modelos tradicionais não trabalham diretamente com imagens, o código extrai informações relevantes como textura, padrões locais e distribuição de intensidade dos pixels.

primeiramente foi feito o carregamento das imagens com a função `load_gray_image():` Observação: Esta etapa pode parecer redundante, pois parte do pré-processamento já foi realizada anteriormente. No entanto, a função foi mantida para melhor organização do pipeline, separação das etapas do processamento e maior clareza na hierarquia das operações do projeto.

GLCM — Características de textura global

A função extract_glcm_features() extrai características usando a Gray Level Co-occurrence Matrix (GLCM), uma técnica que analisa relações entre pixels vizinhos.

Ela:
* reduz os níveis de cinza da imagem para simplificar o cálculo
* calcula relações entre pixels em diferentes direções
* extrai propriedades estatísticas da textura da imagem
  
As características extraídas são:
* contrast → variação de intensidade entre pixels
* correlation → relação entre pixels vizinhos
* energy → uniformidade da textura
* homogeneity → similaridade entre pixels próximos

Essas informações ajudam a identificar padrões estruturais associados à presença de tumores.

LBP — Textura local da imagem

A função extract_lbp_features() utiliza Local Binary Pattern (LBP) para capturar padrões locais de textura.

Ela:
* compara cada pixel com seus vizinhos
* gera padrões binários que representam a textura local
* cria um histograma com a distribuição desses padrões

Isso permite detectar pequenas variações estruturais na imagem.

Histograma de níveis de cinza

A função extract_gray_histogram() calcula a distribuição das intensidades dos pixels da imagem.

Ela:
* mede quantos pixels existem em cada nível de cinza
* normaliza os valores para comparação entre imagens
* representa a aparência geral da imagem

Essa informação descreve a composição global da imagem.

Logo após, os dados são armazenados 
```
brain_mri_train.csv
brain_mri_test.csv
```

---

## classificação

**Arquivo:** `classificacao.py`

Para a ultima etapa desse projeto, é implementado funções importantes que com Validação Cruzada Estratificada, Pipeline de Treinamento e Otimização com GridSearchCV.

Primeiramente, os dados são carregados a partir de arquivos .csv que foram criados da mineração de dados. Com estes datasets, foram cridadas:
* `X_train` e `X_test` → variáveis preditoras (features)
* `y_train` e `y_test` → rótulos das classes (labels)

**Validação Cruzada Estratificada**
A implementação da valização cruzada utiliza 5 splits, ou seja, 5 divisões que garante que:
* a proporção das classes seja mantida em cada divisão
* o modelo seja avaliado de forma mais confiável
* o risco de overfitting seja reduzido

Ou seja, a validação gruzada é uma técnica muito importande para a confiabilidade dos modelos

**Pipeline de Treinamento**
Cada modelo é treinado usando um Pipeline, que organiza o fluxo de processamento:
* Padronização dos dados (StandardScaler)

Normaliza as características para melhorar o desempenho dos modelos.

*Treinamento do modelo de classificação

O algoritmo aprende padrões nos dados para realizar a classificação.

Portando, o uso da pipeline foi utilizada porque evita vazamento de dados e melhora a organização do processo.

**Otimização com GridSearchCV**
GridShearch é uma técnica bastante importante e prática, sento utilizada em todos os modelos

Sua implementação foi realizada em todos os modelos, assim realizando:
* testa diferentes combinações de hiperparâmetros
* seleciona automaticamente a melhor configuração
* utiliza validação cruzada para avaliação
* otimiza o modelo com base na métrica F1-score

Após utilizar o GridShearch, o código avalia os quatro algoritmos de classificação:

Support Vector Machine (SVM)
* busca o melhor limite de separação entre as classes
* testa diferentes kernels e valores de regularização
* utiliza balanceamento de classes

K-Nearest Neighbors (KNN)
* classifica com base nos vizinhos mais próximos
* testa diferentes números de vizinhos e métricas de distância

Regressão Logística
* modelo probabilístico para classificação binária
* ajusta o parâmetro de regularização
* utiliza balanceamento das classes

Random Forest
* conjunto de árvores de decisão
* reduz overfitting e melhora a precisão
* testa profundidade e parâmetros das árvores

Avaliação dos Modelos

Após o treinamento, cada modelo é avaliado usando o conjunto de teste.

São geradas:
* Melhores configurações encontradas
* F1-score médio
* Classification Report (precisão, recall e F1-score)

Matriz de confusão

A matriz de confusão mostra:
* acertos do modelo
* erros de classificação
* desempenho na detecção de tumores

Essa etapa transforma as características extraídas das imagens em previsões de diagnóstico, permitindo comparar diferentes algoritmos e identificar o modelo com melhor desempenho para detecção de tumores cerebrais.

---


## Conclusão 

---

Projeto desenvolvido para fins acadêmicos e aprendizado em Machine Learning.
