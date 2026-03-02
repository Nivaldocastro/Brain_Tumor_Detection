import pandas as pd
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

X = np.load("X.npy")
y = np.load("y.npy")

print(X.shape, y.shape)

# Separação de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Dataset dividido em \n80% Treino: {X_train.shape} \n20% teste: {X_test.shape}")

print("Distribuição das classes no treino:", np.bincount(y_train))
print("Distribuição das classes no teste:", np.bincount(y_test))

# Mineração de dados
def load_gray_image(path, size=(224, 224)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return img

def extract_glcm_features(img):
    img = (img / 16).astype(np.uint8) # Reduz os niveis de cinza

    distances = [1]                            # Distância de 1 pixel
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Direção da relação entre pixels

    glcm = graycomatrix(
        img,
        distances=distances,
        angles=angles,
        levels=16,
        symmetric=True,
        normed=True
    )

    #  Extração das features
    features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        features.extend(graycoprops(glcm, prop).flatten())

    return np.array(features) # Retorna os resultados em forma de vetor

# Textura local da imagem
def extract_lbp_features(img, radius=1, n_points=8):
    lbp = local_binary_pattern(
        img,
        n_points,
        radius,
        method='uniform'
    )

    n_bins = n_points + 2
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    return hist

def extract_gray_histogram(img, bins=32):
    hist = cv2.calcHist([img], [0], None, [bins], [0, 256])
    hist = hist.flatten()
    hist = hist / hist.sum()
    return hist

def extract_features(images):
    features = []
    for img_array in images:
        # Como as imagens já estão normalizadas (0-1) e em 224x224,
        # precisamos converter de volta para uint8 (0-255)
        # conforme esperado pelas funções de GLCM/LBP definidas anteriormente.
        img_uint8 = (img_array * 255).astype(np.uint8)
        img_resized = cv2.resize(img_uint8, (224, 224))

        glcm_feat = extract_glcm_features(img_resized)
        lbp_feat = extract_lbp_features(img_resized)
        hist_feat = extract_gray_histogram(img_resized)

        combined = np.hstack([
            glcm_feat,
            lbp_feat,
            hist_feat
        ])
        features.append(combined)
    return np.array(features)

# Extraindo características usando os arrays já carregados
X_train_feat = extract_features(X_train)
X_test_feat  = extract_features(X_test)

print("Features extraídas com sucesso!")
print("Shape treino:", X_train_feat.shape)
print("Shape teste:", X_test_feat.shape)


# Criando DataFrames a partir das características já extraídas na célula anterior
df_train = pd.DataFrame(X_train_feat)
df_train['label'] = y_train

df_test = pd.DataFrame(X_test_feat)
df_test['label'] = y_test

# Salvando os arquivos CSV
df_train.to_csv('brain_mri_train.csv', index=False)
df_test.to_csv('brain_mri_test.csv', index=False)

print("Arquivos CSV salvos com sucesso!")
print(f"Linhas no treino: {len(df_train)}")
print(f"Linhas no teste: {len(df_test)}")
