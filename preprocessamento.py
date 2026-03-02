import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

caminho_dataset = "brain_tumor_dataset"

X = [] # vai guardar as imagens
y = [] # vai guardar a label de cada imagem


# Redmenciona as imagens em um tamaanho padrão 224x224 sem calsar distorção
def resize_com_padding(img, tamanho=224):
    h, w = img.shape[:2]
    scale = tamanho / max(h, w)

    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    nova_img = np.zeros((tamanho, tamanho), dtype=img.dtype)

    y_offset = (tamanho - new_h) // 2
    x_offset = (tamanho - new_w) // 2

    nova_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

    return nova_img

for label in os.listdir(caminho_dataset):
    pasta_classe = os.path.join(caminho_dataset, label)


    # garantino que realmente é uma pasta
    if os.path.isdir(pasta_classe):
        for arquivo in os.listdir(pasta_classe):
            caminho_img = os.path.join(pasta_classe, arquivo)

            img = cv2.imread(caminho_img)
            if img is None:
                continue

            # transformando em tons de cinza
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Verificando a dimenção das imagens
            if img is not None:
                print(arquivo, img.shape)


            if img is not None:
                img = resize_com_padding(img, 224)

                X.append(img)
                y.append(label)

X = np.array(X)
y = np.array(y)

# normalização
X = X.astype("float32") / 255.0


# labels yes, no = 1, 0
y = np.where(y == "yes", 1, 0)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("labels:", np.unique(y))

print(Counter(y))


# Verificação de imagens
idx = 10

plt.imshow(X[idx].reshape(224, 224), cmap="gray")
plt.title("Tumor" if y[idx] == 1 else "Sem tumor")
plt.axis("off")
plt.show()


# Armazenamento dos dados pré-processados usando Numpy
np.save("X.npy", X)
np.save("y.npy", y)

print("Dados salvos!")