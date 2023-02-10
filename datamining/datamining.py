import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
import os
from pylab import rcParams
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
#Importa minha função
import f_stats_fig as fsf

caminho_planilha = r'C:\Users\DELL\anaconda3\envs\datamining\project\data\serie_temporal.xlsx' #caminho do arquivo

#Cria pasta para salvar as figuras e os gráficos
#Inserir o caminho onde quer salvar as figuras. Edite o trecho:"G:\Meu Drive\2022.1\Monitoria\Modelagem\python\aqui\SantaCecilia_chuva_vazao"
caminho = r'C:\Users\DELL\anaconda3\envs\datamining\project\data\figuras_datamining' 

if not os.path.exists(caminho):
    os.makedirs(caminho)

#Transforma caminho_planilha numa série de dados
sheet_name = "Planilha1"
header = 0
usecols = "A:Q"
engine = "openpyxl"
na_values = ["ALO TEM ALGO ERRADO"]
df = pd.read_excel(io=caminho_planilha,sheet_name=sheet_name,header=header,usecols=usecols,engine=engine,na_values=na_values)

df.columns = df.columns.str.strip() #Retira strings vazias dos dados
print(df.dtypes) #Verifica se as datas estão sendo lidas como datas e se os números decimais estão sendo lidos como números decimais
print(df.describe()) #faz estatísticas iniciais para cada coluna de postos (colunas que possuem float numbers)
print(df)

#Parâmetros
#Coluna do eixo x (No caso é a data)
x = "Data"
#Período para a decomposição temporal
period = 12
#Número de colunas usadas sem contar a data(no caso é postos + Q)
n = 16

#_______________________________________PARTE 1
for i in df.columns[1:]:
    #Séries Temporais
    figuras_SerieTemporal = fsf.serie_temporal(caminho, x, df, i)
    
    #Histogramas de Frequência
    figuras_HistogramaFrequencia = fsf.histograma_frequencia(caminho, df, i)

    #Figuras com as Decomposições Temporais
    figuras_DecomposicaoTemporal = fsf.serie_decomposicaotemporal(caminho, x, df, i, period)

#Gráfico de Dispersão
figura_GraficoDispersao = fsf.serie_temporal_total(caminho, x, df)

#Gráfico Matriz de Correlação
figura_MatrizCorrelacao = fsf.matriz_correlacao(caminho, df)

#Dendograma Clusterizado Hierarquico
figura_DendogramaClusterHierarquico = fsf.dendograma_cluster_hierarquico(caminho, df, x)

#_______________________________________PARTE 2
df_data = df
df_data.set_index(x, inplace=True)

normalizacao = fsf.standard(df)

pca = fsf.pca(normalizacao, n_components=9, tol=0.15, svd_solver='arpack') #Análise das componentes principais

#Porcentagem da variancia explicada para cada componente
variancia = pd.DataFrame(pca[1]['Variância'])
variancia.index = pca[0].columns
variancia.columns = ['Explained variance (%)']
variancia['Explained variance (%)'] = variancia['Explained variance (%)'].apply(lambda x: round(x*100, 2))

#Gráfico da Variancia para cada PCA
figura_variancia_PCA = fsf.grafico_PCA_variancia(caminho, variancia)

#Componentes principais para cada posto
pca_features = pd.DataFrame(pca[1]['Eixo principal por feature']).T
pca_features.index = df_data.columns
pca_features.columns = pca[0].columns

#Grafico PCA
k = 9 #Número de gráficos para cada componente principal
for j in range(1, k+1): 
    figura_grafico_PCA = fsf.grafico_PCA(caminho, pca_features, j)

#Grafico de PCA1 + PCA2 e variancia total explicada entre as duas
figura_grafico_PCA1PCA2 = fsf.grafico_PCA1_PCA2(caminho, variancia, pca_features, n)

#Grafico de PCA1 + PCA2 e variancia total explicada entre as duas
figura_grafico_PCA1PCA2PCA3 = fsf.grafico_PCA1_PCA2_PCA3(variancia, pca_features)

#FIM