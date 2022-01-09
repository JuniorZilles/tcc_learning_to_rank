import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties
from tools import read_json
#from matplotlib import colors


def build_scatter():
    df = pd.read_excel('data/algoritmos_ano_freq.xlsx', header=1)
    df.columns = ["Algoritmo/Modelo", "2005", "2006", "2007", "2008", "2009", "2010",
                  "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2019", "2020", "2021"]
    newdf = df.sort_values("Algoritmo/Modelo", ascending=False)
    x = []
    y = []
    z = []
    for index, row in newdf.iterrows():
        algo = '',
        for ano, qtd in row.iteritems():
            if ano == 'Algoritmo/Modelo':
                algo = qtd
            elif qtd >= 1.0:
                y.append(algo)
                x.append(int(ano))
                z.append(int(qtd))
                
    cmap = plt.cm.viridis
    
    plt.scatter(x, y, c=z, cmap=cmap,  s=100)
    plt.colorbar(ticks=np.linspace(1, 5, 5), label='Quantity')
    plt.xlabel('Year')
    plt.ylabel('Algorithm/Model')
    x.append(2018)
    plt.xticks(x)
    plt.title("Quantity X Algorithm/Model X Year")
    plt.grid(True)
    plt.savefig('alg_ano_arti_x.png', dpi=1920, orientation='portrait')
    plt.show()


def readbuild_histogram():
    df = pd.read_excel('data/algoritmos_ano_freq.xlsx', header=1)
    df.columns = ["Algoritmo/Modelo", "2005", "2006", "2007", "2008", "2009", "2010",
                  "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2019", "2020", "2021"]

    items_g1 = []
    items_g2 = []
    items_g3 = {}
    items_g4 = []
    start = 0
    colors = cm.turbo(np.linspace(0, 1, len(df.index)))
    for index, row in df.iterrows():
        algo = '',
        qtds = []
        soma = 0
        for ano, qtd in row.iteritems():
            if ano == 'Algoritmo/Modelo':
                algo = qtd
            elif qtd >= 1.0:
                items_g1.append([algo, ano, colors[start]])
                items_g2.append([algo, ano, int(qtd)])
                qtds.append(int(qtd))
                soma += qtd
            else:
                qtds.append(0)

        items_g3[algo] = qtds
        items_g4.append([algo, int(soma)])
        start += 1

    items_g1.sort(key=lambda x: x[1])
    items_g2.sort(key=lambda x: x[1])
    items_g4.sort(reverse=True, key=lambda x: x[1])

    fontP = FontProperties()
    fontP.set_size('x-small')

    i= 0
    for y in items_g4:
        plt.bar(y[0], y[1], label=y[0],color=colors[i])
        plt.xticks(y[0], " ")
        i+=1
    plt.ylabel("Quantity")
    plt.title("Quantity of studies X Algorithms/Models")
    plt.grid(True, axis='y')
    plt.legend(title='Algorithms/Models',  loc='upper right', ncol=3, prop=fontP)
    plt.savefig('alg_arti.png', dpi=1920, orientation='portrait')
    plt.show()
    print("fec")

def build_dataset_scatter():
    df = pd.read_excel('data/dataset_ano.xlsx', header=1)
    df.columns = ['Dataset', 'Anos']
    df = df.sort_values("Dataset", ascending=False)
    x = []
    y = []
    z = []
    for index, row in df.iterrows():
        item = {}
        dataset = row['Dataset']
        anos = row['Anos'].split(',')
        item[dataset] = {}
        for ano in anos:
            if ano in item[dataset]:
                item[dataset][ano] += 1
            else:
                item[dataset][ano] = 1
        for key in list(item.keys()):
            for ano in item[key].keys():
                x.append(int(ano))
                y.append(key)
                z.append(item[key][ano])
                
    cmap = plt.cm.viridis
    
    plt.scatter(x, y, c=z, cmap=cmap,  s=100)
    plt.colorbar(ticks=np.linspace(1, 3, 3), label='Quantity')
    plt.xlabel('Year')
    plt.ylabel('Dataset')
    x.append(2014)
    x.append(2015)
    x.append(2017)
    x.append(2018)
    plt.xticks(x)
    plt.title("Quantity X Datasets X Year")
    plt.grid(True)
    plt.savefig('dataset_ano.png', dpi=1920, orientation='portrait')
    plt.show()

def plot_graph(position:int, dataset:str, itens:dict):
    fig, ax = plt.subplots()
    for a in ['rank_ndcg_xgboost', 'rank_xendcg_lgbm', 'lambdarank_lgbm', 'regression_xgboost', 'regression_lgbm']:
        iteration = [x for x in range(0, len(itens[a][f'NDCG@{position}']))]
        ax.plot(iteration, itens[a][f'NDCG@{position}'], label=a)
    #ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0.15, 0.65, 0.05))
    ax.set(xlabel='Iterações', ylabel=f'nDCG@{position}',
        title=f'nDCG@{position} obtido durante o treinamento do {dataset}')
    ax.grid()
    ax.legend()

    fig.savefig(f"images/stript_nDCG{position}_{dataset}.png", dpi=1920, orientation='portrait')
    #plt.show()
def build_line_train_chart():
    data = read_json('train.recover.json')
    for dataset in ['MSLR10K', 'MSLR30K', 'OHSUMED', 'TD2003', 'TD2004']:
        #1,3,5,
        for position in [10]:
            plot_graph(position, dataset, data[dataset])
            
def main():
    #build_scatter()
    #readbuild_histogram()
    #build_dataset_scatter()
    build_line_train_chart()


main()