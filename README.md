# Detecção de Tumores Cerebrais com Machine Learning

Este projeto aplica técnicas de aprendizado de máquina para classificar imagens de ressonância magnética (MRI) em duas categorias: presença ou ausência de tumor cerebral.

Foram avaliados modelos clássicos de classificação — KNN, SVM, Regressão Logística e Random Forest — combinados com técnicas de extração de características de textura (GLCM, LBP e histogramas).

A comparação entre os modelos foi realizada utilizando métricas como F1-score, precisão, recall e matriz de confusão, com foco especial na capacidade de detectar corretamente casos com tumor, considerando a relevância clínica da aplicação.

---

## 📁 Estrutura do Projeto
```
├── preprocessamento.py            # Pré-processamento e correlação
|    ├──── X.npy, y.npy            # Armazenamento dos dados pré-processados 
├── split+data_mining.py           # Separação do treino e teste mais a extração de dados
|    ├──── brain_mri_train.csv     # Armazenamento dos dados coletados da extração
|    ├──── brain_mri_test.csv
├── classificacao.py               # Classificação com GridSearchCV
├── imagens_Brain_Tumor_Detection  # Imagens de resultados
└── README.md
```

---
## 📂 Dataset

**Fonte:** Kaggle

**Link:** https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

**Nome do dataset original:** Brain MRI Images for Brain Tumor Detection

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
Nesta etapa inicial foi realizada a importação do dataset, compreensão da estrutura dos dados e preparação das imagens para a etapa de classificação.

Inicialmente, foram criadas duas listas vazias para armazenar:

- `X`: imagens processadas  
- `y`: rótulos (labels) correspondentes  

```python
X = []  # Armazena as imagens
y = []  # Armazena as labels
```
Em seguida, foi definida a função resize_com_padding, responsável por redimensionar as imagens para o tamanho padrão de 224x224 pixels sem causar distorção.

A escolha do tamanho 224x224 foi baseada na análise da média das dimensões das imagens do dataset, que se aproximavam desse padrão. Além disso, esse tamanho é amplamente utilizado em tarefas de visão computacional.

O pré-processamento incluiu as seguintes etapas:
* Conversão das imagens para escala de cinza utilizando COLOR_BGR2GRAY 
* verificação da dimensão das imagens
* Aplicação do `resize_com_padding`
* normalização para escala de 0 a 1
* Conversão das labels: yes = 1 e no = 0

Para validar o pré-processamento, foi utilizado o `matplotlib` para visualizar imagens após as transformações:
```python
idx = n  # índice de qualquer imagem do dataset

plt.imshow(X[idx].reshape(224, 224), cmap="gray")
plt.title("Tumor" if y[idx] == 1 else "Sem tumor")
plt.axis("off")
plt.show()
```

Essa etapa permite verificar se as imagens:

* Não sofreram distorções
* Não foram cortadas indevidamente
* Mantiveram características importantes
* Preservaram qualidade visual adequada

<img src="/imagens_Brain_tumor_Detection/brain_no.png" alt="Logo" width="400" height="auto">  <img src="/imagens_Brain_tumor_Detection/brain_yes.png" alt="Logo" width="400" height="auto">

Após a validação, os dados processados foram salvos para utilização posterior na etapa de divisão (train/test split) e mineração de dados.


---

## split + data_mining

**Arquivo:** `split+data_mining.py`

Nessa etapa, foi implementado o split (Separação do treino e teste) e mineração de dados

**split**

O dataset foi dividido em 80% para treino (202 amostras) e 20% para teste (51 amostras).

Dois pontos importantes foram considerados:

- Utilização de `random_state=42` para garantir reprodutibilidade dos resultados.
- Uso do parâmetro `stratify=y` para manter a proporção original das classes (aproximadamente 60% com tumor e 40% sem tumor) tanto no conjunto de treino quanto no de teste.
Portanto, ficou
```
Distribuição das classes no treino: [ 78 124]
Distribuição das classes no teste:  [ 20  31]
```
**data_mining**

Essa etapa realiza a mineração de dados das imagens MRI, transformando cada imagem em um conjunto de características numéricas (features) que podem ser utilizadas pelos modelos de machine learning para classificação.

Como modelos tradicionais não trabalham diretamente com imagens, o código extrai informações relevantes como textura, padrões locais e distribuição de intensidade dos pixels.

Primeiramente, foi feito o carregamento das imagens com a função `load_gray_image():` Observação: Esta etapa pode parecer redundante, pois parte do pré-processamento já foi realizada anteriormente. No entanto, a função foi mantida para melhor organização do pipeline, separação das etapas do processamento e maior clareza na hierarquia das operações do projeto.

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

Após a extração das características, os dados foram salvos nos arquivos:

- `brain_mri_train.csv`
- `brain_mri_test.csv`

Essa etapa permite reutilizar os dados minerados sem necessidade de recalcular as features, tornando o pipeline mais eficiente e organizado.

---

## classificação

**Arquivo:** `classificacao.py`

Para a última etapa desse projeto, são implementadas funções como Validação Cruzada Estratificada, Pipeline de Treinamento e Otimização com GridSearchCV.

Primeiramente, os dados são carregados a partir de arquivos .csv que foram criados da mineração de dados. Com estes datasets, foram criadas:
* `X_train` e `X_test` → variáveis preditoras (features)
* `y_train` e `y_test` → rótulos das classes (labels)

**Validação Cruzada Estratificada**
A implementação da validação cruzada utiliza 5 splits, ou seja, 5 divisões que garantem que:
* a proporção das classes seja mantida em cada divisão
* o modelo seja avaliado de forma mais confiável
* o risco de overfitting seja reduzido

Ou seja, a validação cruzada é uma técnica muito importante para a confiabilidade dos modelos

**Pipeline de Treinamento**
Cada modelo é treinado usando um Pipeline, que organiza o fluxo de processamento:
* Padronização dos dados (StandardScaler)

A padronização é aplicada apenas nos dados de treino dentro da validação cruzada, evitando vazamento de dados e garantindo avaliação justa.

*Treinamento do modelo de classificação

O algoritmo aprende padrões nos dados para realizar a classificação.

Portanto, a pipeline foi utilizada porque evita vazamento de dados e melhora a organização do processo.

**Otimização com GridSearchCV**
GridSearch é uma técnica bastante importante e prática, sendo utilizada em todos os modelos

Sua implementação foi realizada em todos os modelos, permitindo:
* testa diferentes combinações de hiperparâmetros
* seleciona automaticamente a melhor configuração
* utiliza validação cruzada para avaliação
* otimiza o modelo com base na métrica F1-score

Observação (O F1-score foi escolhido como métrica principal por equilibrar precisão e recall, sendo adequado para problemas com possível desbalanceamento de classes).

Após utilizar o GridSearch, o código avalia os quatro algoritmos de classificação:

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

---

## Resultados Numéricos

Após o treinamento, cada modelo é avaliado usando o conjunto de teste.

São geradas:
* Melhores configurações encontradas
* F1-score médio
* Classification Report (precisão, recall e F1-score)

Esta tabela mostra os melhores hiperparâmetros e o melhor F1-score Médio (CV) de cada modelo que o GridSearch formou

<img src="/imagens_Brain_tumor_Detection/F1-score_Médio_(CV).png" alt="Logo" width="800" height="auto">

Esta tabela mostra o desempenho geral dos modelos com o KNN mostrando ter um melhor desempenho tanto na Accuracy quanto no F1-score

<img src="/imagens_Brain_tumor_Detection/desempenho_geral.png" alt="Logo" width="800" height="auto">

Esta tabela mostra o desempenho em relação à classe tumor, com o KNN mostrando ter um melhor desempenho tanto na Accuracy quanto no F1-score

<img src="/imagens_Brain_tumor_Detection/desempenho_tumor.png" alt="Logo" width="800" height="auto">

Esta tabela mostra o desempenho em relação à classe sem tumor, com o KNN mostrando ter um melhor desempenho tanto na Accuracy quanto no F1-score

<img src="/imagens_Brain_tumor_Detection/desempenho_sem_tumor.png" alt="Logo" width="800" height="auto">

Portanto, em questão de métricas, o KNN possui as melhores métricas

<img src="/imagens_Brain_tumor_Detection/melhor_modelo.png" alt="Logo" width="800" height="auto">

Porém, como é um dataset de tumores cerebrais, o mais importante é saber sobre a detecção dos tumores, então para ter certeza de qual o melhor modelo, foi implementado uma matriz de confusão para cada modelo

--- 

**Matriz de confusão**

A matriz de confusão mostra:
* acertos do modelo
* erros de classificação
* desempenho na detecção de tumores

Ou seja, a matriz de confusão permite visualizar diretamente falsos positivos e falsos negativos, fornecendo uma análise mais interpretável do comportamento do modelo.
  
<img src="/imagens_Brain_tumor_Detection/mc_svm.png" alt="Logo" width="400" height="auto"> <img src="/imagens_Brain_tumor_Detection/mc_knn.png" alt="Logo" width="400" height="auto">
<img src="/imagens_Brain_tumor_Detection/mc_logisticregression.png" alt="Logo" width="400" height="auto"> <img src="/imagens_Brain_tumor_Detection/mc_randomforest.png" alt="Logo" width="400" height="auto">


## Escolha do Modelo para Aplicação Prática

Embora o KNN tenha apresentado melhor desempenho geral em termos de F1-score médio e acurácia, o modelo SVM foi escolhido como principal para aplicação prática.

Isso ocorre porque o SVM apresentou:

* Maior recall para a classe "Tumor"
* Menor número de falsos negativos
* Melhor sensibilidade na detecção de tumores

No contexto médico, reduzir falsos negativos é essencial, pois deixar de detectar um tumor pode ter consequências graves. Portanto, priorizou-se o modelo com melhor capacidade de detecção da classe positiva.

---

## Conclusão 

Este projeto demonstrou a aplicação prática de técnicas de visão computacional e aprendizado de máquina na detecção de tumores cerebrais a partir de imagens de ressonância magnética.

A combinação de métodos de extração de características baseadas em textura (GLCM, LBP e histogramas) com algoritmos clássicos de classificação mostrou-se eficaz para um dataset de pequeno porte.

Entre os modelos avaliados, o KNN apresentou melhor desempenho geral em métricas como acurácia e F1-score. No entanto, considerando o contexto médico da aplicação, o modelo SVM foi escolhido como principal por apresentar maior recall para a classe "Tumor" e menor número de falsos negativos, priorizando a sensibilidade na detecção de casos positivos.

Os resultados reforçam que abordagens tradicionais de machine learning, quando bem estruturadas e combinadas com técnicas adequadas de extração de características, ainda podem oferecer desempenho competitivo em problemas de classificação de imagens médicas.

Este trabalho também evidencia a importância da escolha criteriosa de métricas de avaliação, especialmente em cenários onde o custo de erros pode ter impacto significativo.

---


## Limitações do Projeto

- Dataset pequeno (253 imagens)
- Ausência de validação externa
- Uso de modelos clássicos ao invés de deep learning

---

## Trabalhos Futuros

Como continuação deste projeto, pretende-se implementar uma abordagem baseada em Deep Learning, utilizando Redes Neurais Convolucionais (CNNs), que são amplamente empregadas em tarefas de classificação de imagens médicas.

A aplicação de modelos como CNNs pode permitir o aprendizado automático de características relevantes diretamente das imagens, reduzindo a dependência de técnicas manuais de extração de atributos.

Além disso, o uso de transfer learning com arquiteturas pré-treinadas poderá contribuir para melhorar a capacidade de generalização do modelo, especialmente considerando o tamanho reduzido do dataset atual.

---

Este projeto possui caráter educacional e exploratório, não devendo ser utilizado para diagnóstico médico real.
