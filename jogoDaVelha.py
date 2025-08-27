import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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


X = df.drop('class', axis=1)
y = df['class']

def preprocessar_dados(baseDeDados):
    # separando os atributos da classe alvo.
    X = baseDeDados.drop('class', axis=1)
    y = baseDeDados['class']

    # transformando atributos categoricos em atributos numéricos, dtype foi utilizado para forçar a utilização de zeros e uns ao inves de true e false.
    X_encoded = pd.get_dummies(X, dtype=int)
    y_encoded = y.map({'positive': 1, 'negative': 0})
    return X_encoded, y_encoded

#aplicando a função de preprocessamento para obter x e y enconded
X_encoded, y_encoded = preprocessar_dados(df)

# Passo 5: Dividir os dados em treino e teste
# test_size=0.3 significa que 30% dos dados irão para o conjunto de teste
# random_state=42 garante que a divisão seja sempre a mesma, para reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42
)
#como ficou os dados depois de separados
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


# modelo 2 : arvore de decisão
print("Iniciando Treinamento : Árvore de Decisão com profundidade 3")
tree_d3 = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_d3.fit(X_train, y_train)
y_pred_tree_d3 = tree_d3.predict(X_test)

print("Iniciando Treinamento : Árvore de Decisão com profundidade 10")
tree_d10 = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_d10.fit(X_train, y_train)
y_pred_tree_d10 = tree_d10.predict(X_test)

imprimir_metricas("Arvore de decisão com profundidade 3", y_test, y_pred_tree_d3)
imprimir_metricas("Árvore de decisão com profundidade 10", y_test, y_pred_tree_d10)


print("Arvore de Decisão categórica")
def calcular_entropia(y):
    """Calcula a entropia de um conjunto de rótulos (alvo)."""
    # Conta a ocorrência de cada classe
    contagem_classes = np.bincount(y)
    # Calcula as probabilidades
    probabilidades = contagem_classes / len(y)

    entropia = 0
    for p in probabilidades:
        if p > 0:
            entropia -= p * np.log2(p)
    return entropia


def calcular_ganho_informacao(X, y, nome_atributo):
    """Calcula o ganho de informação ao dividir os dados por um atributo específico."""
    # Entropia do conjunto de dados original (pai)
    entropia_pai = calcular_entropia(y)

    # Valores únicos no atributo
    valores_unicos = X[nome_atributo].unique()

    # Entropia ponderada dos filhos
    entropia_filhos_ponderada = 0
    total_amostras = len(y)

    for valor in valores_unicos:
        # Pega os índices onde o atributo tem um valor específico
        indices_subconjunto = X[nome_atributo] == valor
        y_subconjunto = y[indices_subconjunto]

        if len(y_subconjunto) == 0:
            continue

        entropia_subconjunto = calcular_entropia(y_subconjunto)
        proporcao_subconjunto = len(y_subconjunto) / total_amostras
        entropia_filhos_ponderada += proporcao_subconjunto * entropia_subconjunto

    return entropia_pai - entropia_filhos_ponderada

# --- Passo 2: Estrutura da Árvore (Nós e a Árvore em si) ---

class Node:
    """Representa um nó na árvore de decisão."""
    def __init__(self, atributo_divisao=None, valor_predito=None):
        self.atributo_divisao = atributo_divisao  # Atributo usado para dividir (ex: 'Tempo')
        self.valor_predito = valor_predito      # Valor da classe se for um nó folha (ex: 'Sim')
        self.filhos = {}                        # Dicionário para os nós filhos (ex: {'Sol': Node(), 'Chuva': Node()})


class ArvoreDecisaoCategorica:
    """Implementação da árvore de decisão para atributos categóricos. (VERSÃO FINAL CORRIGIDA)"""

    def __init__(self, profundidade_max=10):
        self.profundidade_max = profundidade_max
        self.raiz = None

    def fit(self, X, y):
        self.classes, y_encoded = np.unique(y, return_inverse=True)
        self.raiz = self._construir_arvore(X, y_encoded, profundidade=0)

    def _melhor_atributo_divisao(self, X, y):
        melhor_ganho = -1
        melhor_atributo = None
        for atributo in X.columns:
            ganho = calcular_ganho_informacao(X, y, atributo)
            if ganho > melhor_ganho:
                melhor_ganho = ganho
                melhor_atributo = atributo
        return melhor_atributo

    def _construir_arvore(self, X, y, profundidade):
        # Casos de parada (folhas)
        if len(np.unique(y)) == 1:
            return Node(valor_predito=self.classes[y[0]])
        if len(X.columns) == 0 or profundidade >= self.profundidade_max:
            valor_mais_comum = Counter(y).most_common(1)[0][0]
            return Node(valor_predito=self.classes[valor_mais_comum])

        melhor_atributo = self._melhor_atributo_divisao(X, y)
        valor_mais_comum_pai = Counter(y).most_common(1)[0][0]  # Movido para cima

        # Se não houver ganho de informação, cria uma folha
        if melhor_atributo is None:
            return Node(valor_predito=self.classes[valor_mais_comum_pai])

        no = Node(atributo_divisao=melhor_atributo)
        # CORREÇÃO 2: Armazena a previsão majoritária no nó atual
        no.predicao_majoritaria = self.classes[valor_mais_comum_pai]

        folha_fallback = Node(valor_predito=self.classes[valor_mais_comum_pai])

        for valor_unico in X[melhor_atributo].unique():
            indices_subconjunto = X[melhor_atributo] == valor_unico
            X_sub = X[indices_subconjunto].drop(melhor_atributo, axis=1)
            y_sub = y[indices_subconjunto]

            if len(y_sub) == 0:
                no.filhos[valor_unico] = folha_fallback
            else:
                no.filhos[valor_unico] = self._construir_arvore(X_sub, y_sub, profundidade + 1)
        return no

    def predict(self, X_novos):
        return [self._prever_amostra(x, self.raiz) for _, x in X_novos.iterrows()]

    def _prever_amostra(self, x, no):
        if no.valor_predito is not None:
            return no.valor_predito

        valor_atributo = x.get(no.atributo_divisao)  # .get() é mais seguro

        # Se o atributo nem existir no dado de teste (improvável, mas seguro)
        if valor_atributo is None:
            return no.predicao_majoritaria  # Retorna a previsão majoritária do nó atual

        # Se o valor do atributo não tiver um galho correspondente
        if valor_atributo not in no.filhos:
            # CORREÇÃO 2: Em vez de retornar uma string, retorna o "chute inteligente"
            return no.predicao_majoritaria

        proximo_no = no.filhos[valor_atributo]
        return self._prever_amostra(x, proximo_no)


X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42
)
minha_arvore = ArvoreDecisaoCategorica(profundidade_max=3)
minha_arvore.fit(X_treino, y_treino)

minhaArvore2 = ArvoreDecisaoCategorica(profundidade_max=10)
minhaArvore2.fit(X_treino, y_treino)

previsoes = minha_arvore.predict(X_teste)
previsao2 = minhaArvore2.predict(X_teste)

imprimir_metricas("Arvore de decisão com profundidade 3 versao 2", y_teste, previsoes)
imprimir_metricas("Árvore de decisao com profundidade 10 versão 2", y_teste, previsao2)




#modelo 3 Random Forest
print("\n Iniciando treinamento : Random Forest com 10 árvores")

rf_10 = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=10)
rf_10.fit(X_train, y_train)
y_pred_rf_10 = rf_10.predict(X_test)

print("Iniciando treinamento : Random Forest com 100 arvores")

rf_100 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=100)
rf_100.fit(X_train, y_train)
y_pred_rf_100 = rf_100.predict(X_test)

imprimir_metricas("Random Forest com 10 árvores", y_test, y_pred_rf_10)
imprimir_metricas("Random Forest com 100 arvores", y_test, y_pred_rf_100)








