from sklearn.datasets import load_wine
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dados_vinho = load_wine()
df = pd.DataFrame(dados_vinho.data, columns=dados_vinho.feature_names)
df['Result'] = dados_vinho.target

for i in df.columns:
    if df[i].dtype in (np.float64, np.int32):
        df[i].fillna(0)
    else:
        df[i].fillna('Na')

x = np.asanyarray(df[['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280/od315_of_diluted_wines','proline']].values)
y = np.asanyarray(df[['Result']].values)
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.3,random_state=28)

def Populacao(Tam_pop):
    populacao = []
    for i in range(0,Tam_pop):
        individuo = []
        #Escolha aleatória do critério utilizado na árvore
        criterio = random.choice(['gini','entropy','log_loss'])
        individuo.append(criterio)

        #Escolha aleatória de max_depth da árvore
        max_depth = random.randint(1,15)
        individuo.append(max_depth)

        #Escolha aleatória no mínimo de amostras
        min_samples_leaf = random.randint(2,20)
        individuo.append(min_samples_leaf)

        #Esolha aleatória do mínimo de amostras
        min_samples_split = random.randint(2,20)
        individuo.append(min_samples_split)

        populacao.append(individuo)
    return populacao

def Mutacao(individuo):
    # Mutação na profundidade máxima
    if random.random() < taxa_mutacao:
        max_depth = random.randint(1, 15)
        individuo[1] = max_depth
    
    # Mutação no critério
    if random.random() < taxa_mutacao:
        criterio = random.choice(['gini', 'entropy','log_loss'])
        individuo[0] = criterio
    
    # Mutação no número mínimo de amostras na folha
    if random.random() < taxa_mutacao:
        min_samples_leaf = random.randint(2, 20)
        individuo[2] = min_samples_leaf

    # Mutação no número mínimo de amostras para divisão
    if random.random() < taxa_mutacao:
        min_samples_split = random.randint(2, 20)
        individuo[3] = min_samples_split

    return individuo

def Selecao(populacao):
    populacao_ordenada = sorted(populacao, key = lambda x : Fitness(x),reverse = True)
    pais_selecionados_para_cruzamento = populacao_ordenada[:len(populacao_ordenada)//2]
    return pais_selecionados_para_cruzamento

def Fitness(individuo):
    tree = DecisionTreeClassifier(max_depth = individuo[1], criterion= individuo[0], min_samples_leaf= individuo[2], min_samples_split= individuo[3])
    tree.fit(x_treino, y_treino)
    y_pred = tree.predict(x_teste)
    accuracy = accuracy_score(y_teste, y_pred)
    return accuracy

def Cruzamento(pai1, pai2):
    # Criando filhos
    filho1 = []
    filho2 = []
    
    # Adicionando o critério ao filho 1 e ao filho 2
    filho1.append(pai1[0])
    filho2.append(pai2[0])

    # Adicionando o max_depth ao filho 1 e ao filho 2
    filho1.append(pai1[1])
    filho2.append(pai2[1])
    
    for i in range(2, len(pai1)):
        if i % 2 == 0:
            filho1.append(pai1[i])
            filho2.append(pai2[i])
        else:
            filho1.append(pai2[i])
            filho2.append(pai1[i])
    
    return filho1, filho2

#Definição de parâmetros
Tam_pop = 50
geracoes = 20
taxa_mutacao = 0.28

#Definição do Algorítimo genético
populacao = Populacao(Tam_pop)
melhores_acuracias = []
for i in range(geracoes):
    proxima_geracao = []
    while len(proxima_geracao) < Tam_pop:
        pai1, pai2 = random.sample(Selecao(populacao),2)
        filho1, filho2 = Cruzamento(pai1, pai2)
        filho1 = Mutacao(filho1)
        filho2 = Mutacao(filho2)
        proxima_geracao.append(filho1)
        proxima_geracao.append(filho2)
    populacao = proxima_geracao
    melhor_da_geracao = max(populacao, key = Fitness)
    melhores_acuracias.append(Fitness(melhor_da_geracao))

#Melhor solução
melhor_s = max(populacao, key = lambda x : Fitness(x))
tree = DecisionTreeClassifier(max_depth = melhor_s[1], criterion= melhor_s[0], min_samples_leaf= melhor_s[2], min_samples_split= melhor_s[3])
tree.fit(x_treino, y_treino)
y_pred = tree.predict(x_teste)
accuracy = accuracy_score(y_teste, y_pred)
print(f'Acurácia gerada pelo algorítimo genético {accuracy}')
print(f'Foi utilizado o critério: {melhor_s[0]}')
print(f'Foi utilizado como max_depth: {melhor_s[1]}')
print(f'Foi utilizado como min_samples_leaf: {melhor_s[2]}')
print(f'Foi utilizado como min_samples_split: {melhor_s[3]}')

kf = KFold(n_splits= 5, random_state= 28, shuffle=True)
tree = DecisionTreeClassifier(criterion=melhor_s[0], max_depth=melhor_s[1], min_samples_leaf=melhor_s[2], min_samples_split=melhor_s[3])
scores = cross_val_score(tree, x, y, cv=kf, scoring='accuracy')

# Resultados
print(f'Média dos scores de cada divisão: {scores.mean()}')
print(f'Variabilidade da acurácia entre todas as dobras: {scores.std()}')
