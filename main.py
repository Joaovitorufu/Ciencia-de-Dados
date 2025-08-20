import pandas as pd

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


#separando os atributos da classe alvo.
X = df.drop('class', axis=1)
y = df['class']

#transformando atributos categoricos em atributos numéricos, dtype foi utilizado para forçar a utilização de zeros e uns ao inves de true e false.
X_encoded = pd.get_dummies(X, dtype = int)
print(X_encoded.head())

y_encoded = y.map({'positive': 1, 'negative': 0})
# Exibindo as primeiras linhas do resultado para verificar
print("--- Alvo (y) após a codificação ---")
print(y_encoded.head())



