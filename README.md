# Projeto Detecção de tumores serebrais com machine learning

Este projeto tem como objetivo aplicar modelos como KNN (K-Nearest Neighbors ou K-Vizinhos Mais Próximos), SVM (Support Vector Machine), Logistic regression e Random Forest em um dataset sobre imagens de exames de ressonância magnética afim de treinalos e posteriormente análisar qaul modelo é melhor comparando não só a acuracy, mas também o desenpenho em relação aos predicts sobre as imagens com tumor.

---

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

