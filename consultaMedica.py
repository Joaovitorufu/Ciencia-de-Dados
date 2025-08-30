import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix

df = pd.read_csv('./Bases de dados/KaggleV2-May-2016.csv', header=0)

def imprimirMetricas(nome_modelo, y_test_verdadeiro, y_previsao):
    acc = accuracy_score(y_test_verdadeiro, y_previsao)
    prec = precision_score(y_test_verdadeiro, y_previsao)
    rec = recall_score(y_test_verdadeiro, y_previsao)
    cm = confusion_matrix(y_test_verdadeiro, y_previsao)

    print(f"\nResultados para {nome_modelo}:")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Revocação: {rec:.4f}")
    print(f"Matriz de Confusão:\n{cm}")

def salvar_matriz_confusao(nome_modelo, y_test_verdadeiro, y_previsao , cmap='Blues'):
    cm = confusion_matrix(y_test_verdadeiro, y_previsao)
    plt.figure(figsize=[8,6])
    sns.heatmap(cm, annot=True, cmap=cmap, fmt='d',
                xticklabels=['Compareceu (0)', 'Não Compareceu (1)'],
                yticklabels=['Compareceu (0)', 'Não Compareceu (1)'])
    plt.xlabel('Previsão do Modelo')
    plt.ylabel('Valor Real')
    plt.title(f'Matriz de Confusão - {nome_modelo}')

    filename = f'matriz_confusao_{nome_modelo.replace(" ","_").lower()}.png'
    plt.savefig(filename)
    print(f"Gráfico da matriz de confusão salvo como '{filename}'")
    plt.close()

#Fase de Pré processamento

#verificando se existe atributos invalidos.
print(df['Age'].max())
print(df['Age'].min())

#removendo as idades invalidas
df['Age'] = df['Age'].clip(lower = 0)

#removendo atributos de controle de index
df=df.drop(columns=['PatientId','AppointmentID'])

#mapeando a coluna gender para atributos numéricos e mapeando a coluna alvo (No-Show)
df['Gender'] = df['Gender'].map({'F':0,'M':1})
df['No-show'] = df['No-show'].map({'No':0, 'Yes':1})

#retirando o Timezone
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.tz_localize(None)
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.tz_localize(None)

#juntando as 2 colunas scheduledDay e AppointementDay em uma coluna unica WaitingTime(tempo de espera) medido e dias
df['WaitingTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

#foi necessário essa correção por conta da operação com datas, AppointmentDay por default é setado como meia-noite(00:00:00) resultando em uma data negativa quando é no mesmo dia das consultas.
#existem alguns valores inconsistentes(negativos) que podem ser resultado de lançamentos errados, esses valores vão ser arredondados para 0.
df['WaitingTime'] = df['WaitingTime'].clip(lower = 0)

#retirando as colunas que foram mescladas em uma nova coluna WaitingTime.
df = df.drop(columns=['ScheduledDay','AppointmentDay'])

#aplicando One-Hot Encoding na coluna Neighbourhood
df= pd.get_dummies(df, columns = ['Neighbourhood'], drop_first = True)

#Selecionando uma amostragem de 5000 amostras do DataFrame já pré-processado.
df_sample = df.sample(n=5000, random_state=42)
print(f"Tamanho do DataFrame após amostragem: {df_sample.shape}")

X = df_sample.drop('No-show', axis = 1)
y = df_sample['No-show']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\nTamanhos dos conjuntos de dados:")
print(f"Features de Treino (X_train): {X_train.shape}")
print(f"Alvo de Treino (y_train):    {y_train.shape}")
print(f"Features de Teste (X_test):   {X_test.shape}")
print(f"Alvo de Teste (y_test):      {y_test.shape}")



#Passo 2 -> Treinar os modelos
#escalonar as features foi necessário para dar pesos iguais as colunas para achar os vizinhos mais proximos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#KNN com k = 3
knn_k3 = KNeighborsClassifier(n_neighbors = 3)
knn_k3.fit(X_train_scaled, y_train)
y_pred_k3 = knn_k3.predict(X_test_scaled)
imprimirMetricas("k-NN com k=3", y_test, y_pred_k3)
salvar_matriz_confusao("k-NN com k=3 consultaMedica", y_test, y_pred_k3, cmap='Blues')

#KNN com k = 7
knn_k7 = KNeighborsClassifier(n_neighbors = 7)
knn_k7.fit(X_train_scaled, y_train)
y_pred_k7 = knn_k7.predict(X_test_scaled)
imprimirMetricas("k-NN com k=7", y_test, y_pred_k7)
salvar_matriz_confusao("k-NN com k=7 consultaMedica", y_test, y_pred_k7, cmap='Blues')


#modelo 2 arvore de decisão balanceada
print("Iniciando Treinamento : Árvore de Decisão com profundidade 3")
tree_d3 = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
tree_d3.fit(X_train, y_train)
y_pred_tree_d3 = tree_d3.predict(X_test)
imprimirMetricas("Arvore de decisão com profundidade 3", y_test, y_pred_tree_d3)
salvar_matriz_confusao("Arvore de decisão com profundidade 3 consultaMedica", y_test, y_pred_tree_d3, cmap='Blues')

print("Iniciando Treinamento : Árvore de Decisão com profundidade 10")
tree_d10 = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
tree_d10.fit(X_train, y_train)
y_pred_tree_d10 = tree_d10.predict(X_test)
imprimirMetricas("Arvore de decisão com profundidade 10", y_test, y_pred_tree_d10)
salvar_matriz_confusao("Arvore de decisão com profundidade 10 consultaMedica", y_test, y_pred_tree_d10, cmap='Blues')

print("\n Iniciando treinamento : Random Forest com 10 árvores")

rf_10 = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=10)
rf_10.fit(X_train, y_train)
y_pred_rf_10 = rf_10.predict(X_test)
imprimirMetricas("Random Forest com 10 arvores", y_test, y_pred_rf_10)
salvar_matriz_confusao("Random Forest com 10 arvores consultaMedica", y_test, y_pred_rf_10, cmap='Blues')

print("\n Iniciando treinamento : Random Forest com 100 árvores")
rf_100 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=100)
rf_100.fit(X_train, y_train)
y_pred_rf_100 = rf_100.predict(X_test)
imprimirMetricas("Random Forest com 100 arvores", y_test, y_pred_rf_100)
salvar_matriz_confusao("Random Fores com 100 arvores consultaMedica", y_test, y_pred_rf_100, cmap='Blues')

