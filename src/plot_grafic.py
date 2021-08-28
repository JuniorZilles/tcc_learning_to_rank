import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties

def read():
    df = pd.read_excel('data/algoritmos_ano_freq.xlsx', header=1)
    df.columns = ["Algoritmo/Modelo","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2019","2020","2021"]
    
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
            if ano  == 'Algoritmo/Modelo':
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
        start+=1     
    
    items_g1.sort(key=lambda x: x[1])
    items_g2.sort(key=lambda x: x[1])
    items_g4.sort(reverse=True,key=lambda x: x[1]) 
    

    # for y in items_g1:
    #     plt.scatter(y[1], y[0], color=y[2])
    # plt.xlabel("Ano")
    # plt.ylabel("Algoritmo/Modelo")
    # plt.title("Algoritmos/Modelos por ano")
    # plt.grid(True)
    # plt.savefig('alg_ano.png', dpi=1920, orientation='portrait')
    # plt.show()

    dfg = pd.DataFrame(items_g2, columns=['Algoritmo/Modelo', 'Ano', 'Quantidade'])
    ax2 = dfg.plot.scatter(x='Ano',
                      y='Algoritmo/Modelo',
                      c='Quantidade',
                      colormap='viridis')
    plt.title("Quantidade X Algoritmos/Modelos X Ano")
    plt.grid(True)
    plt.savefig('alg_ano_arti.png', dpi=1920, orientation='portrait')
    plt.show()

    fontP = FontProperties()
    fontP.set_size('x-small')

    # dfl = pd.DataFrame(items_g3, index=["2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2019","2020","2021"])
    # ax3 = dfl.plot.line()
    # plt.title("Quantidade X Algoritmos/Modelos X Ano")
    # plt.xlabel("Ano")
    # plt.ylabel("Quantidade")
    # plt.legend(title='Algoritmos/Modelos', bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
    # plt.grid(True)
    # plt.savefig('alg_ano_arti_line.png', dpi=1920, orientation='portrait')
    # plt.show()

   

    i= 0
    for y in items_g4:
        plt.bar(y[0], y[1], label=y[0],color=colors[i])
        plt.xticks(y[0], " ")
        i+=1
    plt.ylabel("Quantidade")
    plt.title("Quantidade de trabalhos X Algoritmos/Modelos")  
    plt.grid(True, axis='y')
    plt.legend(title='Algoritmos/Modelos',  loc='upper right', ncol=3, prop=fontP)
    plt.savefig('alg_arti.png', dpi=1920, orientation='portrait')
    plt.show()
    print("fec")


def main():
    
    read()


main()