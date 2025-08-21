import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

#nomes das colunas com base no arquivo .names.
column_names = [
    'top-left-square',
    'top-middle-square',
    'top-right-square',
    'middle-left-square',
    'middle-middle-square',
    'middle-right-square',
    'bottom-left-square',
    'bottom-middle-square',
    'bottom-right-square',
    'class'
]

df = pd.read_csv('./Bases de dados/tic+tac+toe+endgame/tic-tac-toe.csv', header=None, names=column_names)

def preprocessar_dados(baseDeDados):
    # separando os atributos da classe alvo.
    X = baseDeDados.drop('class', axis=1)
    y = baseDeDados['class']

    # transformando atributos categoricos em atributos numéricos, dtype foi utilizado para forçar a utilização de zeros e uns ao inves de true e false.
    X_encoded = pd.get_dummies(X, dtype=int)
    # print(X_encoded.head())

    y_encoded = y.map({'positive': 1, 'negative': 0})
    # Exibindo as primeiras linhas do resultado para verificar
    # print("--- Alvo (y) após a codificação ---")
    # print(y_encoded.head())
    return X_encoded, y_encoded

#aplicando a função de preprocessamento para obter x e y enconded
X_encoded, y_encoded = preprocessar_dados(df)


# Passo 5: Dividir os dados em treino e teste
# test_size=0.3 significa que 30% dos dados irão para o conjunto de teste
# random_state=42 garante que a divisão seja sempre a mesma, para reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42
)

print(f"Formato de X_train (treino): {X_train.shape}")
print(f"Formato de y_train (treino): {y_train.shape}")
print(f"Formato de X_test (teste):  {X_test.shape}")
print(f"Formato de y_test (teste):  {y_test.shape}")

# --- Modelo 1: k-NN com k=3 ---
print("--- Iniciando Treinamento: k-NN (k=3) ---")

#Algoritmo KNN com o K =3 e K =7
knn_k3 = KNeighborsClassifier(n_neighbors=3)
knn_k7 = KNeighborsClassifier(n_neighbors=7)

#Treinar o modelo
knn_k3.fit(X_train, y_train)
knn_k7.fit(X_train, y_train)
# fazendo previsões nos dados de teste

y_pred_knn_k3 = knn_k3.predict(X_test)
y_pred_knn_k7 = knn_k7.predict(X_test)

def imprimir_metricas(nome_modelo, y_test_verdadeiro, y_previsao):
    acc = accuracy_score(y_test_verdadeiro, y_previsao)
    prec = precision_score(y_test_verdadeiro, y_previsao)
    rec = recall_score(y_test_verdadeiro, y_previsao)

    print(f"\nResultados para {nome_modelo}:")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Revocação: {rec:.4f}")

imprimir_metricas("k-NN com k=3", y_test, y_pred_knn_k3)
imprimir_metricas("k-NN com k=7", y_test, y_pred_knn_k7)


