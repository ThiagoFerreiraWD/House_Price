#!/usr/bin/env python
# coding: utf-8

# ![bannerHousePrice.png](attachment:bannerHousePrice.png)

# ![image.png](attachment:image.png)

# Para o presente projeto fora selecionado o *Dataset* contendo diversas características que descrevem uma casa, como número de vagas na garagem, tamanho do lote, material do telhado entre outras, cada registro contido na base de treinamento detalha não só as características do imóvel e seu processo de venda, bem como o valor de venda do mesmo.
# &nbsp;
# 
# Os dados foram selecionados a partir de uma competição disponível no site do [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

# ![image.png](attachment:image.png)

# Obter o melhor resultado *(quanto menor, melhor)* para a métrica utilizada na competição, descrita abaixo:
# &nbsp;
# 
# **- RMSE (Root Mean Squarred Error):** Medida que calcula a "Raiz Quadrada Média" dos erros entre todos os valores observados e as predições. Tem como característica penalizar os erros com maior grau.
# Eis a sua fórmula:
# &nbsp;
# 
# &nbsp;
# 
# ![image.png](attachment:image.png)
# &nbsp;
# 
# &nbsp;
# 
# 
# Para tal optei por rodar diversos algoritmos de regressão (sem utilização da biblioteca Pycaret), comparando seu resultados e selecionando o com o melhor desempenho, para posterior aplicação aos dados de validação e submissão ao [site da competição](https://www.kaggle.com).

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[1]:


import pandas as pd
pd.set_option('display.max_columns', 100), pd.set_option('display.max_rows', 50)
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ![image.png](attachment:image.png)

# In[2]:


df_treino = pd.read_csv('Arquivos/train.csv')
df_teste = pd.read_csv('Arquivos/test.csv')
df_combinado = pd.concat([df_treino, df_teste])


# ![image.png](attachment:image.png)

# In[3]:


# Exibição das primeiras linhas do DF
df_combinado.head(2)


# In[4]:


# Criação de função para posterior averiguação do tamanho das bases.
def colLin(df, t):
    print(f'\033[1m{t}\033[0m possui \033[1m{df.shape[0]} registros\033[0m com \033[1m{df.shape[1]} atributos cada\033[0m.')

# DF Treino x DF Teste
colLin(df_treino, 'DataFrame Treino'), colLin(df_teste, 'DataFrame Teste');


# In[5]:


# Informações Básicas das Colunas do DF.
df_combinado.info()


# In[6]:


# Exibição descritiva das variáveis numéricas do DF (38 no total).
qtdLinhas = str(df_combinado.describe().shape[1]) + ' atributos numéricos no DataFrame.'
display(qtdLinhas, df_combinado.describe())


# In[7]:


# Exibição descritiva das variáveis categóricas do DF (43 no total).
qtdLinhas = str(df_combinado.describe(include='O').shape[1]) + ' atributos categóricos no DataFrame.'
display(qtdLinhas, df_combinado.describe(include='O'))


# *** 
# Para essa aplicação não utilizaremos as variáveis categóricas no treinamento dos modelos.
# ***

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[8]:


# Plotagem do gráfico de correlação entre as variáveis.
df_corr = df_treino.drop('Id', axis=1)
plt.figure(figsize=(30, 20))
plt.title('Gráfico 1 - Correlação entre as Features', pad=30, fontsize=25)
sns.heatmap(df_corr.corr(),
            fmt='.1g',
            annot=True,
            square=False,
            cbar=False,
            linewidths=0.25,
            cmap='gray_r',
            annot_kws={'size': 12})
plt.show()


# ***
# **Gráfico 1 - Observações**
# 
# * Podemos observar que há algumas *features* com alta correlação com a varíavel *target*, como 'OverallQual', 'TotalBSmtSF' e 'GrLivArea' por exemplo. Veremos as TOP 10 à seguir.
# * Trabalharemos com as 10 variáveis com maior relação para a construção dos nossos modelos.
# ***

# In[9]:


# Plotagem do gráfico das 10 features com maior correlação com a variável target ('SalePrice').
k = 10
plt.figure(figsize=(30, 10))
plt.title('Gráfico 2 - Correlação entre as TOP Features', pad=30, fontsize=25)
cols = df_treino.corr().nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_treino[cols].values.T)
hm = sns.heatmap(cm,
                 cbar=False,
                 annot=True,
                 square=False,
                 fmt='.2f',
                 annot_kws={'size': 25},
                 cmap='gray_r',
                 yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()


# ***
# **Gráfico 2 - Observações**
# 
# * As *features* 'OverallQual', 'GrLivArea' e 'GarageCars' estão fortemente correlacionadas com a *feature* 'SalePrice';
# * As *features* 'GarageCars' e 'GarageArea' possuem correlação semelhantes, uma vez que o tamanho da garagem geralmente corresponde à quantidade total de vagas de carros disponíveis, observa-se a correlação entre elas (0.88); Veremos essa correlação mais a fundo adiante;
# * De igual modo as *features* ['TotalBsmtSF' e '1stFlrSF'] e ['GrLivArea' e 'TotRmsAbvGrd'] possuem uma alta correlação entre si.
# ***

# ![image.png](attachment:image.png)

# In[10]:


# Descrição da varíavel target.
df_treino.SalePrice.describe()


# In[11]:


# Plotagem do histograma da variável target.
fig, ax = plt.subplots(figsize=(30, 6))
sns.distplot(df_treino.SalePrice, color="black", bins=150)
plt.title('Gráfico 3 - Distribuição dos Valores', pad=30, fontsize=25)

ax.set_xlim(df_treino.SalePrice.min(),df_treino.SalePrice.max())
ax.set_xticks(range(0, df_treino.SalePrice.max(), 25000))
ax.set_ylabel('')
ax.tick_params(axis='y', labelleft=False, left=None)

for axis in ['top', 'right', 'left']:
    ax.spines[axis].set_color(None)
ax.spines['bottom'].set_linewidth(1.75)

plt.show()


# In[12]:


print(f'\033[1mAssimetria Positiva: {round(df_treino.SalePrice.skew(), 3)}.')
print(f'\033[1mCurtóise Platicúrtica: {round(df_treino.SalePrice.kurt(), 3)}.')


# In[13]:


features = ['OverallQual', 'GrLivArea', 'GarageArea']

figura = plt.figure(figsize=(30, 5))
figura.suptitle('Gráfico 4 - Correlação entre as Features com a SalePrice', fontsize=25, y=1.1)

for i in range(3):
    figura.add_subplot(1, 3, i + 1)
    plt.scatter(x=df_treino[features[i]], y=df_treino['SalePrice'], c='gray')
    plt.title(f'Gráfico 4.{i+1}. {features[i]}', loc='left', fontsize=20)


# ***
# **Gráfico 4. Observações**
# 
# Podemos observar alguns ***outliers***, principalmente no Gráfico 4.2.
# 
# ***

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[14]:


# Criação do DF com os Valores Nulos
df_nulos = df_combinado.drop('SalePrice', axis=1)
df_nulos = df_nulos.isnull().sum().sort_values(ascending=False)[0:34].to_frame().reset_index()
df_nulos.columns = ['Feature', 'Count']

# Plotagem do gráfico
fig, ax = plt.subplots(figsize=(30, 6))
sns.barplot(x='Feature', y='Count', data=df_nulos[0:16], palette='gray')
plt.title('Gráfico 5. % NaN por Features', fontsize=25, pad=30)

ax.set_ylabel('')
ax.set_xlabel('')
ax.tick_params(axis='y', labelleft=False, left=None)

for axis in ['top', 'right', 'left']:
    ax.spines[axis].set_color(None)
ax.spines['bottom'].set_linewidth(2.5)

for i in ax.patches:
    ax.annotate(round(i.get_height()/len(df_combinado)*100,2),       
                (i.get_x() + i.get_width() / 2, i.get_height() + 25),
                ha='center',
                va='baseline',
                xytext=(0, 1), textcoords='offset points',
                fontsize=20)


# ***
# **Gráfico 5. Observações 1**
# 
# * **PoolQC:** Pool quality;
# * **MiscFeature:** Miscellaneous feature not covered in other categories;
# * **Alley:** Type of alley access do property;
# * **Fence:** Fence quality; e
# * **FireplaceQu:** Fireplace quality.
# 
# Possuem uma quantidade de valores ausentes demasiadamente alta, portanto iremos excluir essas *features* do *DataFrame*.
# 
# ***
# 

# In[15]:


# Exclusão das colunas mencionadas acima, e da coluna ID
features_excluir = df_nulos.Feature[0:5].values
df_combinado.drop(features_excluir, axis=1, inplace=True)
df_combinado.drop('Id', axis=1, inplace=True)
shape = df_combinado.shape
print(f'Agora o DataFrame possui \033[1m{shape[0]} linhas\033[0m por \033[1m{shape[1]} colunas\033[0m.')


# ***
# **Gráfico 5. Observações 2**
# 
# Os atributos **GarageFinish**, **GarageQual**, **GarageCond** e **GarageYrBlt** possuem a mesma porcentagem de valores ausentes, bem como uma alta relação entre si. Vamos analisá-las mais profundamente.
# 
# ***

# ![image.png](attachment:image.png)

# In[16]:


# Selecionando todas as colunas que possuam a palavra 'Garage'
features_garagem = [coluna for coluna in df_combinado.columns if 'garage' in coluna.lower()]

# Criação de um DataFrame apenas com as colunas selecionadas acima
df_garagem = df_combinado[features_garagem]
df_garagem['SalePrice'] = df_combinado.SalePrice
df_garagem.head(2)


# In[17]:


plt.figure(figsize=(30, 5))
plt.title("Gráfico 6. Correlação Features 'Garage'", fontsize=25, pad=30)

sns.heatmap(df_garagem.corr(), 
            fmt='.1g',
            annot=True,
            square=False,
            cbar=False,
            linewidths=0.25,
            cmap='gray_r',
            annot_kws={'size': 10})
plt.show()


# ***
# **Gráfico 6. Observações**
# 
# As *features* **GarageArea** e **GarageCars** possuem uma alta correlação, uma vez que é justificável que quanto maior a área da garagem, maior será a quantidade de vagas para carros. Iremos optar apenas por uma delas, no caso será a **GarageCars**.
# 
# ***

# ![image.png](attachment:image.png)

# In[18]:


# Plotagem Gráfico Outliers 'GrLivArea'
plt.figure(figsize=(30, 5.5))
plt.scatter(df_treino.GrLivArea, df_treino.SalePrice, c='gray')
plt.title("Gráfico 7. Verificando Outliers da Feature 'GrLivArea'", fontsize=25, pad=30)
plt.show()


# In[19]:


# Exibição das linhas identificadas com Outliers
outliers_grlivarea = df_treino[(df_treino.GrLivArea > 4000) & (df_treino.SalePrice < 300000)]
outliers_grlivarea


# In[20]:


# Exclusão das linhas com Outliers identificados
df_combinado = df_combinado.reset_index()
df_combinado.drop([523, 1298], inplace=True)


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[21]:


# Separação dos DFs Treino e de Validação
df_treino = df_combinado[df_combinado.SalePrice.notnull()]
df_valida = df_combinado[df_combinado.SalePrice.isnull()]


# In[22]:


# Divisão do DF nas 'variáveis' preditoras e target
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
X = df_treino[features]
y = df_treino.SalePrice


# In[23]:


# Divisão das 'variáveis' entre treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)


# In[24]:


colLin(X_treino, 'X_treino'), colLin(X_teste, 'X_teste');


# ![image.png](attachment:image.png)

# In[25]:


# Criação das variáveis escalonadas
scalerX = StandardScaler()
X_treino_scaled = scalerX.fit_transform(X_treino)
X_teste_scaled = scalerX.fit_transform(X_teste)

scalerY = StandardScaler()
y_treino_scaled = scalerY.fit_transform(y_treino.to_frame())
y_teste_scaled = scalerY.fit_transform(y_teste.to_frame())


# ![image.png](attachment:image.png)

# In[26]:


# Preparação
scalerX = StandardScaler()
X_scaled = scalerX.fit_transform(X)
scalerY = StandardScaler()
y_scaled = scalerY.fit_transform(y.to_frame())

regressores = []
regressores.append(('Random Forest', RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)))
regressores.append(('SVM', SVR(kernel='rbf')))
regressores.append(('Redes Neurais', MLPRegressor(max_iter=1000, hidden_layer_sizes=(6, 6))))
regressores.append(('Gradient Boosting', GradientBoostingRegressor(learning_rate=0.06, n_estimators=100, max_depth=3)))
regressores.append(('Ada Boost', AdaBoostRegressor(n_estimators=500, random_state=0)))
regressores.append(('XGBoost', XGBRegressor(booster='dart', n_estimators=300, max_depth=3, learning_rate=0.01, random_state=0, n_jobs=-1)))
resultados = []

for modelo, reg in regressores:
    print(f'Treinando -> {modelo} ...')
    for i in tqdm(range(30)):
        kfold = KFold(n_splits=10, shuffle=True, random_state=i)                    
        scores = cross_val_score(reg, X_scaled, y_scaled, cv=kfold)
        temp = [modelo, scores.mean()]
        resultados.append(temp)


# In[27]:


# Transformação da lista de resultados em um DataFrame
colunas = ['Random Forest', 'SVM', 'Redes Neurais', 'Gradient Boosting', 'Ada Boost', 'XG Boost']
df_resultados = pd.DataFrame()

for coluna in colunas:
    df_resultados[coluna] = 0    
    
for i in range(30):
    df_resultados.loc[i] = [resultados[i][1], resultados[i + 30][1], 
                            resultados[i + 60][1], resultados[i + 90][1],
                            resultados[i + 120][1],resultados[i + 150][1]]

display(df_resultados.head(2), df_resultados.describe())


# In[28]:


# Exportação do DF criado
df_resultados.to_csv('Arquivos/resultadosWithKFold.csv')


# In[29]:


# Criação do Dicionario com os Algoritmos e a  Média do Desempenho
dictResults = {coluna: df_resultados[coluna].mean() for coluna in df_resultados.columns}


# In[30]:


# Plotagem do Gráfico de Desempenho dos Algoritmos
x = [i for i in sorted(dictResults, key = dictResults.get)]
y = [dictResults[i] for i in sorted(dictResults, key = dictResults.get)]

fig, ax = plt.subplots(figsize=(30, 5))
sns.barplot(y, x, palette='Greys')
ax.tick_params(axis='x', labelbottom=False), ax.tick_params(axis='y', labelsize=15)
ax.set_xlabel(''), ax.set_ylabel(''), ax.set_title('Gráfico 8. Score Médio dos Algoritmos', fontsize=25, pad=30)
ax.spines['left'].set_linewidth(2.5)

for axis in ['top', 'right', 'bottom']:    ax.spines[axis].set_color(None)

for i, v in enumerate(y):
    plt.text(0.01, i+0.2 , s=(str(round(v*100,3)) + '%'), color='white', fontsize=25)


# ***
# Após aplicação da técnica de Validação Cruzada, podemos observar que o algoritmo que obteve o melhor desempenho foi o Gradient Boosting.

# ![image.png](attachment:image.png)

# In[31]:


features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

X = df_treino[features]
y = df_treino.SalePrice

scalerX = StandardScaler()
X_scaled = scalerX.fit_transform(X)
scalerY = StandardScaler()
y_scaled = scalerY.fit_transform(y.to_frame())

best_model = GradientBoostingRegressor(learning_rate=0.06, n_estimators=100, max_depth=3)
best_model.fit(X_scaled, y_scaled)


# ![image.png](attachment:image.png)

# In[32]:


# Substituição dos Valores NaN na Base de Validação
df_teste.GarageCars.fillna(df_teste.GarageCars.mean(), inplace=True)
df_teste.TotalBsmtSF.fillna(df_teste.TotalBsmtSF.mean(), inplace=True)

X = df_teste[features]
X_scaled = scalerX.fit_transform(X)

previsao = best_model.predict(X_scaled)
previsao_final = scalerY.inverse_transform(previsao.reshape(-1, 1))


# In[33]:


price = pd.DataFrame(previsao_final)
price.rename(columns={0:'SalePrice'}, inplace=True)
df_sub = pd.DataFrame({'Id': df_teste.Id, 'SalePrice': np.round(price.SalePrice, 0)})
df_sub.to_csv('Arquivos/GradientBoosting.csv', index=False)


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ***
# Após submissão ao Kaggle, eis o resultado obtido. Lembrando que a métrica utilizada para avaliar o algoritmo foi o RMSE (Root Mean Squarred Error).
# ***

# ![image.png](attachment:image.png)

# |                 **Github**                	|                 **Linkedin**                 	|                  **Kaggle**                  	|               **Email**              	|
# |:-----------------------------------------:	|:--------------------------------------------:	|:--------------------------------------------:	|:------------------------------------:	|
# | [![image.png](attachment:image.png)](https://www.github.com/ThiagoFerreiraWD) 	| [![image-3.png](attachment:image-3.png)](https://www.linkedin.com/in/tferreirasilva) 	| [![image-4.png](attachment:image-4.png)](https://www.kaggle.com/thiagoferreirasilva) 	| [![image-2.png](attachment:image-2.png)](mailto:thiago.ferreirawd@gmail.com) 	|  

# ![image.png](attachment:image.png)
