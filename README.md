# semana17inteligencia


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Gerando os dados
np.random.seed(42)

dados = pd.DataFrame({
    "horas_estudo": np.random.randint(1, 6, 50),
    "exercicios": np.random.randint(10, 100, 50),
    "frequencia": np.random.randint(60, 100, 50),
    "participacao": np.random.randint(1, 10, 50),
})

dados["nota_anterior"] = dados["horas_estudo"] * 1.5 + np.random.normal(0, 1, 50)

# Etapa 1 - Matriz de correlação
print("Matriz de correlação:")
print(dados.corr().round(2))

# Etapa 2 - Redução de colinearidade
# Horas de estudo e nota anterior possuem alta correlação.
# Vamos manter nota_anterior e remover horas_estudo.

dados_reduzidos = dados.drop(columns=["horas_estudo"])

print("\nDados após remover a variável correlacionada:")
print(dados_reduzidos.head())

# Etapa 3 - Criação de nova feature
dados_reduzidos["engajamento_total"] = (
    dados_reduzidos["frequencia"] +
    dados_reduzidos["participacao"]
)

print("\nNova feature criada:")
print(dados_reduzidos[["frequencia", "participacao", "engajamento_total"]].head())

# Etapa 4 - Aplicação do PCA
X = dados_reduzidos.drop(columns=["engajamento_total"])

# Padronização dos dados
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X)

# Aplicando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_padronizado)

print("\nDados transformados pelo PCA:")
print(X_pca[:5])

print("\nVariância explicada por cada componente:")
print(pca.explained_variance_ratio_.round(3))

print("\nVariância explicada acumulada:")
print(pca.explained_variance_ratio_.sum().round(3))


Relatório  Seleção e Criação de Features

Nessa atividade analisamos alguns dados de estudantes, como horas de estudo, exercícios, frequência, participação e nota anterior.

Primeiro fizemos uma matriz de correlação para descobrir quais variáveis tinham relação entre si. Percebemos que horas de estudo e nota anterior tinham uma correlação alta.

Para diminuir a repetição de informações, escolhemos retirar horas de estudo e manter a nota anterior. Depois criamos uma nova variável chamada engajamento_total, juntando frequência e participação.

Também aplicamos o PCA para diminuir a quantidade de variáveis e manter as informações mais importantes.

A seleção de features ajuda o modelo a trabalhar melhor com os dados e pode melhorar sua generalização. A colinearidade pode atrapalhar porque algumas variáveis acabam trazendo informações muito parecidas.

No final, o PCA deixou os dados mais simples, mas também dificultou um pouco a interpretação das variáveis. Como não treinamos um modelo de previsão, não foi possível comparar exatamente o desempenho antes e depois.
