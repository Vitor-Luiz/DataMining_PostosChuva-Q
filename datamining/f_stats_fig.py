import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from pylab import rcParams
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
import os

#Normaliza um DataFrame
def standard(df, **scaler_kw):

    array_normalizado = StandardScaler(**scaler_kw).fit_transform(df)
    df_normalizado = pd.DataFrame(array_normalizado, columns = df.columns, index = df.index)

    return df_normalizado

#Definição das Componentes Principais
def pca(df, **pca_kw):

    #Calcula as componentes principais de df
    dfs = []  
    
    n_components = pca_kw.get('n_components', len(df.columns))
    
    inicializacao = PCA(n_components = n_components)
    pca_ = inicializacao.fit_transform(df)

    for i in range(n_components):
        dfs.append(pd.DataFrame(pca_[:,i],
                                index = df.index,
                                columns = [f"PC{i + 1}"]))
        
    df_final = pd.concat(dfs, axis = 1)
    
    return [df_final,
            {"Eixo principal por feature": inicializacao.components_, 
            "Variância": inicializacao.explained_variance_ratio_, 
            "Média por feature": inicializacao.mean_, 
            "Número estimado de componentes": inicializacao.n_components_,
            "Variância residual": inicializacao.noise_variance_}]

#Função para criação do gráfico com os valores em cima das barras
def autolabel(bars, ax: plt.axes) -> None:

    for bar in bars:
        height = bar.get_height()
        x = height
        y = round(x, 1)
        ax.annotate(f'{y} %',
                    xy=(bar.get_x() + bar.get_width() / 1.8, y),
                    xytext=(0, 15),
                    textcoords='offset points',
                    ha='center', 
                    va='bottom', 
                    fontsize=12, 
                    color='black', 
                    rotation=0, 
                    weight='bold')

#SÉRIE TEMPORAL
def serie_temporal(caminho, x, df, i):

    #Cria pasta para salvar as figuras e os gráficos
    if not os.path.exists(f"{caminho}\SerieTemporal"):
        os.makedirs(f"{caminho}\SerieTemporal")

    #Plota os gráficos das séries temporais
    plt.figure()
    plt.plot(df[x], df[i])
    plt.title(i)
    plt.ylabel('Valor')
    plt.xlabel(x)
    plt.savefig(f"{caminho}\SerieTemporal\{i}.png")
    figura = plt.show()
    return figura

#HISTOGRAMA DE FREQUÊNCIA 
def histograma_frequencia(caminho, df, i):

    #Cria pasta para salvar as figuras e os gráficos
    if not os.path.exists(f"{caminho}\Frequencia"):
        os.makedirs(f"{caminho}\Frequencia")

    plt.figure()
    plt.hist(df[i], bins = 20)
    plt.title(i)
    plt.ylabel('Frequência')
    plt.xlabel('Valor')
    plt.savefig(f"{caminho}\Frequencia\{i}.png")
    figura = plt.show()
    return figura

#SÉRIE TEMPORAL COMPLETA
def serie_temporal_total(caminho, x, df):

    if not os.path.exists(f"{caminho}\Grafico_Dispersao"):
        os.makedirs(f"{caminho}\Grafico_dispersao")
#Série temporal inteira
    plt.rcParams["figure.figsize"] = (15,8) #controla o tamanho da figura
    for i in df.columns[1:]:
        plt.scatter(df[x], df[i], label=i)
    plt.ylabel('Valor')
    plt.xlabel(x)
    plt.title('Série Temporal')
    plt.legend(bbox_to_anchor = (1, 1))
    plt.savefig(f'{caminho}\Grafico_Dispersao\Grafico_Dispersao.png')
    figura = plt.show()

    return figura

#DECOMPOSIÇÃO TEMPORAL
def serie_decomposicaotemporal(caminho, x, df, i, period):

    if not os.path.exists(f"{caminho}\decomposicao_temporal"):
        os.makedirs(f"{caminho}\decomposicao_temporal")
    
    #Série temporal feito acima.
    df_data = df.set_index(x)
    decomposition = sm.tsa.seasonal_decompose(df_data[i], model = 'additive', period=period)
    figuras = decomposition.plot()
    plt.savefig(f'{caminho}/Decomposicao_Temporal/{i}.png')
    figura = plt.show()

    return figura

#MATRIZ DE CORRELAÇÃO
def matriz_correlacao(caminho, df):

    if not os.path.exists(f"{caminho}\matriz_correlacao"):
        os.makedirs(f"{caminho}\matriz_correlacao")
    
    #Matriz de correlação entre as colunas com float numbers
    matrix = df.corr()

    #Plota a matriz de correlação
    plt.rcParams["figure.figsize"] = (10,7)
    sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='RdBu')
    plt.savefig(f'{caminho}\matriz_correlacao\matriz_correlacao.png')
    figura = plt.show()
    
    return figura

#DENDOGRAMA CLUSTERIZADO Hierarquico
def dendograma_cluster_hierarquico(caminho, df, x):

    ms = df.set_index(x)
    ms = ms.T
    diss_matrix = squareform(pdist(ms, 'euclidean'))

    if not os.path.exists(f"{caminho}\Dendograma_Cluster_Hierarquico"):
        os.makedirs(f"{caminho}\Dendograma_Cluster_Hierarquico")
    
    z = hierarchy.linkage(diss_matrix, optimal_ordering=True)

    plt.figure()
    plt.rcParams["figure.figsize"] = (16,10)
    plt.xlabel('Postos')
    plt.ylabel('Height')
    plt.title('Dendograma Clusterizado')
    dn = hierarchy.dendrogram(z, labels=ms.index)
    plt.savefig(f'{caminho}\Dendograma_Cluster_Hierarquico\Dendograma_Cluster_Hierarquico.png')
    figura = plt.show()
    
    return figura

#Gráfico que usa o autolabel com a porcentagem das variancias associadas a cada componente principal
def grafico_PCA_variancia(caminho, variancia):

    if not os.path.exists(f"{caminho}\PCA_variancia"):
        os.makedirs(f"{caminho}\PCA_variancia")

    #Variancia explicada por Componente Principal
    with plt.style.context('ggplot'):
        fig3, ax = plt.subplots(figsize=(18, 8))
        fig3.suptitle(f'Variancia explicada por Componente Principal', style='oblique', weight='bold', fontsize=15)
        bars = ax.bar(variancia.index, variancia['Explained variance (%)'])
        autolabel(bars, ax)
        ax.set_ylabel('Variance (%)', fontsize=15, labelpad=20)
    figura = plt.savefig(f"{caminho}\PCA_variancia\PCA_variancia.png")
    
    return figura

#Valor da componente principal x para cada posto
def grafico_PCA(caminho, pca_features, j):
    
    if not os.path.exists(f"{caminho}\PCA"):
        os.makedirs(f"{caminho}\PCA")
    
    fig20, ax = plt.subplots(figsize=(10, 5))

    fig20.suptitle(f'PC{j}', weight='bold', fontsize=20)

    fig20.tight_layout()

    pc1 = ax.scatter(pca_features.index, 
                     pca_features.loc[:, f'PC{j}'], 
                     s=100)

    ax.set_xlabel('Postos', fontsize=15, labelpad=20)
    
    figura = str(j)
    figura = 'PC'+figura
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f'{caminho}\PCA\{figura}.png')
    figuras = plt.show()

    return figuras

#Gráfico de PCA1 e PCA2
def grafico_PCA1_PCA2(caminho, variancia, pca_features, n):

    if not os.path.exists(f"{caminho}\PC1_PC2"):
        os.makedirs(f"{caminho}\PC1_PC2")

    variancia2 = variancia.iloc[:2,:]

    total1 = variancia2.sum()
    total1 = float(total1)
    plt.figure(figsize=(12,8))
    sns.scatterplot(x="PC1", y="PC2", hue=pca_features.index, palette=sns.color_palette("hls", n),
                    data=pca_features, legend="full", alpha=1, s=100)
    plt.legend(bbox_to_anchor = (1, 1))
    plt.title(f'Variancia total explicada entre PC1 e PC2: {total1:.2f}%')
    plt.savefig(f"{caminho}\PC1_PC2\PC1_PC2.png")

    figura = plt.show()

    return figura

#Gráfico em 3D de PCA1, PCA2 e PCA3
def grafico_PCA1_PCA2_PCA3(variancia, pca_features):
    
    #Gráfico em 3D interativo
    variancia3 = variancia.iloc[:3,:2]
    total = variancia3.sum()
    total = float(total)
    new=pca_features.assign(Postos=pca_features.index)
    figura = px.scatter_3d(new, x='PC3', y='PC2', z='PC1', color=new['Postos'],            
                           title=f'Variancia total explicada: {total:.2f}%', labels={'PC1': 'PC1',
                                                                                     'PC2': 'PC2',
                                                                                     'PC3': 'PC3'})
    figura = figura.show()

    return figura

